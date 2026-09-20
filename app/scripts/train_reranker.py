import pandas as pd
from catboost import CatBoostClassifier, Pool
from app.domain.services.feature_extractor import FeatureExtractor
from app.infrastucture.config import DATA_PROCESSED, RERANKER_MODEL_FILE, SEED

print(">> Загрузка выборок")
train = pd.read_parquet(DATA_PROCESSED / "trainset.parquet")
evalset = pd.read_parquet(DATA_PROCESSED / "evalset.parquet")
print(f"train: {len(train)} строк, {train['group_id'].nunique()} групп")
print(f"eval:  {len(evalset)} строк, {evalset['group_id'].nunique()} групп")

DROP = ["pool_size", "n_candidates", "e5_found", "frida_found", "bm25_found"]
FEATURES = [f for f in FeatureExtractor.FEATURE_NAMES if f not in DROP]
print(f"Признаков: {len(FEATURES)}")

train = train.sort_values("group_id").reset_index(drop=True)
evalset = evalset.sort_values("group_id").reset_index(drop=True)

train_pool = Pool(train[FEATURES], train["label"], group_id=train["group_id"])
eval_pool = Pool(evalset[FEATURES], evalset["label"], group_id=evalset["group_id"])

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_state=SEED,
    early_stopping_rounds=100,
    verbose=100,
    task_type="CPU",
    thread_count=-1
)

print("\n >> Обучение")
model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
RERANKER_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(RERANKER_MODEL_FILE))
print(f"\n>> Модель сохранена: {RERANKER_MODEL_FILE}")
print(f">> Лучшая итерация: {model.get_best_iteration()}")

print("\n >> Важность признаков:")
imp = pd.Series(model.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
print(imp.to_string())