from app.pipeline.pipeline import Pipeline
from app.domain.services.trainset_builder import TrainsetBuilder
from app.preprocessing.splitting import make_eval_set
from app.infrastucture.config import DATA_PROCESSED, SEED
import pandas as pd

N_QUERIES = 40_000

pipeline = Pipeline()
train = pipeline._setup()

fit_queries, fit_gold = make_eval_set(train, "fit", eligible_only=False)
rank_queries, rank_gold = make_eval_set(train, "rank", eligible_only=False)
print(f"fit всего:  {len(fit_queries)}")
print(f"rank всего: {len(rank_queries)}")

sampled = fit_queries.sample(n=min(N_QUERIES, len(fit_queries)), random_state=SEED)

builder = TrainsetBuilder(pipeline)
builder.build(sampled, dict(zip(fit_gold["qid"], fit_gold["item_ids"])),
              DATA_PROCESSED / "trainset.parquet")
print(f"\n=== Eval-выборка: {len(rank_queries)} запросов ===")
builder.build(rank_queries, dict(zip(rank_gold["qid"], rank_gold["item_ids"])),
              DATA_PROCESSED / "evalset.parquet")