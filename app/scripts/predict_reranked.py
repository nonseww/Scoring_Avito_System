"""Финальный ответ на бенчмарк: RRF + CatBoost-реранк верхушки.

Схема та же, что на val (eval_reranker_head):
  4 источника → RRF → первые TOP_N кандидатов → CatBoost
  (reranker_top150.cbm, обучен на верхушке RRF с весами на сэмплинг)
  → топ-50. Хвост после TOP_N остаётся в порядке RRF.
val: 0.8466 против 0.8231 у чистого RRF.

Ответ пишется в answers_rerank.csv; answers.csv (попытка 2, RRF)
не трогаем — он нужен для сравнения.

Запуск из корня проекта:
    python -u -m app.scripts.predict_reranked
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from app.infrastucture.config import (ANSWER_FILE, BENCHMARK_Q_FILE,
                                      RERANKER_MODEL_FILE, ROOT)
from app.infrastucture.loading import load_bench_queries
from app.pipeline.pipeline import Pipeline
# Те же функции и константы, что при оценке на val, — чтобы режимы не разъехались
from app.scripts.eval_reranker_head import FEATURES, TOP_N, rerank_head, row_item_ids

MODEL_FILE = RERANKER_MODEL_FILE.with_name("reranker_top150.cbm")
OUT_FILE = ROOT / "answers_rerank.csv"


def predict(pipeline: Pipeline, model: CatBoostClassifier) -> dict:
    """Прогоняет запросы бенчмарка, возвращает {query_id: [до 50 item_id]}."""
    queries = load_bench_queries(BENCHMARK_Q_FILE)
    assert queries["query_id"].is_unique, "query_id в бенчмарке повторяются"
    texts = queries["normed_search_query"].tolist()

    print(">> Подкатегории и кодирование запросов")
    # with_proba: вероятности нужны для признаков (microcat_proba и др.)
    microcats, probas = pipeline.microcat_service.predict_microcats_with_proba(texts)
    qvs_e5 = pipeline.e5_service.encode_queries(texts)
    qvs_frida = pipeline.frida_service.encode_queries(texts)

    print(f">> Прогон по бенчмарку: {len(queries)} запросов")
    t0 = time.time()
    predictions = {}
    n_fallback = 0

    for i, row in enumerate(queries.itertuples()):
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

        # Пустой список кандидатов (пустой пул) — реранкать нечего
        if not head:
            predictions[row.query_id] = rrf_order[:50]
            n_fallback += 1
            continue

        # Как при обучении и на val: признаки по верхушке,
        # реальный размер списка — через n_candidates_total
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
        ids = row_item_ids(X, head)  # защита от сдвига индекса
        scores = model.predict_proba(X[FEATURES])[:, 1]
        predictions[row.query_id] = rerank_head(
            rrf_order, dict(zip(ids, scores)), TOP_N
        )[:50]

        if i % 250 == 0 and i > 0:
            speed = (i + 1) / (time.time() - t0)
            eta = (len(queries) - i - 1) / speed / 60
            print(f"  {i}/{len(queries)}, {speed:.1f} зпр/с, осталось ~{eta:.0f} мин")

    print(f"time = {(time.time() - t0) / 60:.0f} мин, без реранка (пустые): {n_fallback}")
    return predictions, set(queries["query_id"])


def check_answer(path, expected_qids: set, bench_ids: set) -> None:
    """Проверки формата сохранённого файла — читаем его заново,
    как будет читать проверяющая система."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["query_id", "answer"], f"колонки: {list(df.columns)}"
    assert len(df) == len(expected_qids), f"строк {len(df)}, ожидаем {len(expected_qids)}"
    assert set(df["query_id"]) == {str(q) for q in expected_qids}, "набор query_id не совпадает"

    lists = df["answer"].str.split()
    sizes = lists.str.len()
    assert (sizes <= 50).all(), "есть ответы длиннее 50"
    assert lists.apply(lambda x: len(x) == len(set(x))).all(), "дубликаты id в ответе"
    bench_str = {str(b) for b in bench_ids}
    alien = sum(1 for lst in lists for it in lst if it not in bench_str)
    assert alien == 0, f"{alien} id не из корпуса бенчмарка"

    print(f"Формат ОК: {len(df)} строк, длина ответа медиана {sizes.median():.0f}, "
          f"мин {sizes.min()}, пустых {(sizes == 0).sum()}")


def compare_with_rrf(new_path, old_path) -> None:
    """Пересечение с ответом попытки 2 (чистый RRF).

    Реранк меняет только часть позиций, поэтому ожидаем ~85–95%.
    Сильно меньше — значит что-то сломалось (например, сдвиг id).
    """
    if not old_path.exists():
        print(f"{old_path} нет — сравнение пропущено")
        return
    new = pd.read_csv(new_path, dtype=str, keep_default_na=False).set_index("query_id")["answer"]
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False).set_index("query_id")["answer"]
    shares = []
    for qid, a in old.items():
        o = set(a.split())
        if o and qid in new.index:
            shares.append(len(o & set(new[qid].split())) / len(o))
    print(f"Пересечение с RRF-ответом: среднее {np.mean(shares):.3f}, "
          f"мин {np.min(shares):.3f}")


def main():
    pipeline = Pipeline()
    pipeline._setup()

    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    print(f"Модель: {MODEL_FILE}")

    predictions, qids = predict(pipeline, model)
    pipeline.evaluation_service.save_answer(predictions, OUT_FILE)
    print(f">> Сохранено в {OUT_FILE}")

    check_answer(OUT_FILE, qids, pipeline.bench_ids)
    compare_with_rrf(OUT_FILE, ANSWER_FILE)


if __name__ == "__main__":
    main()