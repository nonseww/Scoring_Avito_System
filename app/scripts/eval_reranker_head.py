import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from app.domain.services.feature_extractor import FeatureExtractor
from app.infrastucture.config import RERANKER_MODEL_FILE, VAL_HEAD_FEATURES_FILE

CACHE = VAL_HEAD_FEATURES_FILE

TOP_N = 150 # сколько верхних кандидатов RRF кэшируем
TOP_RERANK_GRID = [75, 100, 150] # все значения <= TOP_N

# Признаки должны совпадать с тем, на чём обучалась оцениваемая модель
DROP = ["e5_found", "frida_found", "bm25_found"]
FEATURES = [f for f in FeatureExtractor.FEATURE_NAMES if f not in DROP]


def row_item_ids(X, candidates):
    """Оценка реранкера на верхушке RRF (val).
    Реранкер переставляет только первые TOP_N кандидатов RRF, хвост остаётся
    в порядке RRF. Первый запуск прогоняет источники и кэширует признаки
    верхушки на диск. Повторные запуски (другая модель) берут их из кэша"""

    ids = X["item_id"].tolist() if "item_id" in X.columns else X.index.tolist()
    assert len(ids) == len(candidates), f"строк {len(ids)}, кандидатов {len(candidates)}"
    assert set(ids) == set(candidates), "id строк X не совпадают с кандидатами"
    return ids


def recall_at_k(preds, gold, k):
    """Макро-recall@k. При k=50 должен совпасть со штатным recall (0.8231 у RRF)."""
    vals = []
    for qid, gold_items in gold.items():
        gold_set = set(gold_items)
        if not gold_set:
            continue
        vals.append(len(set(preds.get(qid, [])[:k]) & gold_set) / len(gold_set))
    return float(np.mean(vals))


def rerank_head(rrf_order, score_by_id, top_rerank):
    """Сортирует первые top_rerank кандидатов по скору модели, хвост не трогает"""
    head, tail = rrf_order[:top_rerank], rrf_order[top_rerank:]
    return sorted(head, key=lambda it: -score_by_id.get(it, -1.0)) + tail


def build_cache():
    """Прогоняет источники по val и сохраняет признаки верхушки RRF."""
    # Импорт здесь: при наличии кэша тяжёлый пайплайн не поднимаем
    from app.pipeline.pipeline import Pipeline
    from app.preprocessing.splitting import make_eval_set

    pipeline = Pipeline()
    train = pipeline._setup()
    val_queries, val_gold = make_eval_set(train, "val")
    gold = dict(zip(val_gold["qid"], val_gold["item_ids"]))
    print(f"Запросов для оценки: {len(val_queries)}")

    texts = val_queries["normed_search_query"].tolist()
    microcats, probas = pipeline.microcat_service.predict_microcats_with_proba(texts)
    qvs_e5 = pipeline.e5_service.encode_queries(texts)
    qvs_frida = pipeline.frida_service.encode_queries(texts)

    rrf_full, union, parts = {}, {}, []
    t0 = time.time()
    for i, row in enumerate(val_queries.itertuples()):
        out = pipeline._run_sources(
            row.normed_search_query, qvs_e5[i], qvs_frida[i], microcats[i],
            row.search_location_id,
            is_delivery=(row.search_is_delivery_search == 1),
            restrict_to=pipeline.bench_ids,
            top_k=500,
        )
        candidates = out["candidates"]
        rrf_order = list(out["rankings"]["rrf"])
        head = rrf_order[:TOP_N]
        rrf_full[row.qid] = rrf_order
        union[row.qid] = candidates

        # Как в TrainsetBuilder: признаки по подмножеству кандидатов,
        # а общий размер списка передаётся отдельно — иначе n_candidates
        # станет 150 вместо реальных ~700
        X = pipeline.feature_extractor.extract(
            candidates=head,
            source_rankings=out["rankings"],
            source_scores=out["scores"],
            query_location_id=row.search_location_id,
            predicted_microcats=microcats[i],
            microcat_probas=probas[i],
            pool_size=len(out["pool"]),
            query_text=row.normed_search_query,
            n_candidates_total=len(candidates),
        )
        ids = row_item_ids(X, head)
        X = X.assign(item_id=ids, qid=row.qid).reset_index(drop=True)
        parts.append(X)

        if i % 250 == 0 and i > 0:
            speed = (i + 1) / (time.time() - t0)
            eta = (len(val_queries) - i - 1) / speed / 60
            print(f"  {i}/{len(val_queries)}, {speed:.1f} зпр/с, осталось ~{eta:.0f} мин")

    feats = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"gold": gold, "rrf_full": rrf_full, "union": union, "feats": feats}, f)
    print(f"Кэш сохранён: {CACHE}, {len(feats)} строк, {(time.time()-t0)/60:.0f} мин")


def evaluate(model_path):
    with open(CACHE, "rb") as f:
        c = pickle.load(f)
    gold, rrf_full, union, feats = c["gold"], c["rrf_full"], c["union"], c["feats"]

    # Потолок: реранк верхушки N может достать не больше Recall@N(RRF)
    for k in (50, 75, 100, 150):
        print(f"RRF Recall@{k}: {recall_at_k(rrf_full, gold, k):.4f}")
    print(f"Покрытие:      {recall_at_k(union, gold, 10**9):.4f}")
    print("(RRF@50 должен быть 0.8231 — иначе что-то разъехалось)\n")

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    feats = feats.assign(_score=model.predict_proba(feats[FEATURES])[:, 1])
    score_by_q = {q: dict(zip(g["item_id"], g["_score"])) for q, g in feats.groupby("qid")}

    print(f"Модель: {model_path}")
    for top in TOP_RERANK_GRID:
        preds = {q: rerank_head(rrf_full[q], score_by_q.get(q, {}), top)[:50]
                 for q in rrf_full}
        print(f"CatBoost top={top:>3} @50: {recall_at_k(preds, gold, 50):.4f}")


if __name__ == "__main__":
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RERANKER_MODEL_FILE
    if not CACHE.exists():
        build_cache()
    evaluate(model_path)