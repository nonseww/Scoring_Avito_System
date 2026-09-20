"""Оценка реранкера на val — сравнение с RRF."""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from app.domain.services.feature_extractor import FeatureExtractor
from app.infrastucture.config import RERANKER_MODEL_FILE
from app.pipeline.pipeline import Pipeline, recall_no_cutoff
from app.preprocessing.splitting import make_eval_set

pipeline = Pipeline()
train = pipeline._setup()

val_queries, val_gold = make_eval_set(train, "val")
gold_dict = dict(zip(val_gold["qid"], val_gold["item_ids"]))
print(f"Запросов для оценки: {len(val_queries)}")

model = CatBoostClassifier()
model.load_model(str(RERANKER_MODEL_FILE))
DROP = ["pool_size", "n_candidates", "e5_found", "frida_found", "bm25_found"]
FEATURES = [f for f in FeatureExtractor.FEATURE_NAMES if f not in DROP]
print(f"Признаков: {len(FEATURES)}")

texts = val_queries["normed_search_query"].tolist()
microcats, probas = pipeline.microcat_service.predict_microcats_with_proba(texts)
qvs_e5 = pipeline.e5_service.encode_queries(texts)
qvs_frida = pipeline.frida_service.encode_queries(texts)

print(">> Прогон по val")
t0 = time.time()
rrf_preds = {}
catboost_preds = {}
union = {}

for i, row in enumerate(val_queries.itertuples()):
    out = pipeline._run_sources(
        row.normed_search_query, qvs_e5[i], qvs_frida[i], microcats[i],
        row.search_location_id,
        is_delivery=(row.search_is_delivery_search == 1),
        restrict_to=pipeline.bench_ids,
        top_k=500,
    )
    candidates = out["candidates"]
    rrf_preds[row.qid] = out["rankings"]["rrf"][:50]
    union[row.qid] = candidates

    X = pipeline.feature_extractor.extract(
        candidates=candidates,
        source_rankings=out["rankings"],
        source_scores=out["scores"],
        query_location_id=row.search_location_id,
        predicted_microcats=microcats[i],
        microcat_probas=probas[i],
        pool_size=len(out["pool"]),
        query_text=row.normed_search_query,
    )
    scores = model.predict_proba(X[FEATURES])[:, 1]
    order = np.argsort(-scores)[:50]
    catboost_preds[row.qid] = [candidates[j] for j in order]

print(f"time = {time.time() - t0:.0f} секунд\n")

print(f"RRF@50:       {pipeline.evaluation_service.recall(rrf_preds, gold_dict):.4f}")
print(f"CatBoost@50:  {pipeline.evaluation_service.recall(catboost_preds, gold_dict):.4f}")
print(f"Покрытие:     {recall_no_cutoff(union, gold_dict):.4f}")