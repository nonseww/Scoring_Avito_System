from app.pipeline.pipeline import Pipeline
from app.domain.services.trainset_builder import TrainsetBuilder
from app.preprocessing.splitting import make_eval_set
from app.infrastucture.config import DATA_PROCESSED, SEED

MAX_QUERIES = 25_000
pipeline = Pipeline()
train = pipeline._setup()

fit_queries, fit_gold = make_eval_set(train, "fit")
rank_queries, rank_gold = make_eval_set(train, "rank")
print(f"fit eligible:  {len(fit_queries)}")
print(f"rank eligible: {len(rank_queries)}")

if len(fit_queries) > MAX_QUERIES:
    fit_queries = fit_queries.sample(n=MAX_QUERIES, random_state=SEED)
    print(f"сэмплировано до: {len(fit_queries)}")

builder = TrainsetBuilder(pipeline)

print(f"\n=== Основная выборка: {len(fit_queries)} запросов ===")
builder.build(fit_queries, dict(zip(fit_gold["qid"], fit_gold["item_ids"])),
              DATA_PROCESSED / "trainset_v2.parquet", batch_size=1000)

print(f"\n=== Eval-выборка: {len(rank_queries)} запросов ===")
builder.build(rank_queries, dict(zip(rank_gold["qid"], rank_gold["item_ids"])),
              DATA_PROCESSED / "evalset_v2.parquet", batch_size=1000)