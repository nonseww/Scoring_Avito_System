import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from app.domain.services.feature_extractor import FeatureExtractor
from app.infrastucture.config import DATA_PROCESSED, RERANKER_MODEL_FILE, SEED

"""Переобучение реранкера на верхушке RRF без пересборки выборки.

Из готовых выборок оставляем кандидатов с глобальным RRF-рангом в первых TOP_N.
Так обучающее распределение совпадает с инференсом, где модель
переранжирует только первые TOP_N кандидатов RRF"""

TOP_N = 150 # совпадает с TOP_N кэша в eval_reranker_head
RANK_COL = "rrf_rank" # глобальный ранг кандидата в RRF
N_HARD = 15
TRAINSET_FILES = [
    DATA_PROCESSED / "trainset_v2.parquet", # fit + половина rank
    DATA_PROCESSED / "evalset_v2.parquet", # вторая половина rank (не val)
]
DROP = ["e5_found", "frida_found", "bm25_found"]
FEATURES = [f for f in FeatureExtractor.FEATURE_NAMES if f not in DROP]
OUT_MODEL = RERANKER_MODEL_FILE.with_name(f"reranker_top{TOP_N}.cbm")


def load_trainsets() -> pd.DataFrame:
    """Читает и объединяет выборки, проверяя, что запросы не пересекаются."""
    parts = [pd.read_parquet(p, engine="fastparquet") for p in TRAINSET_FILES]
    for p, d in zip(TRAINSET_FILES, parts):
        print(f"{p.name}: {len(d)} строк, {d['group_id'].nunique()} групп")
    overlap = set(parts[0]["group_id"]) & set(parts[1]["group_id"])
    assert not overlap, f"общих групп: {len(overlap)} — выборки пересекаются"
    return pd.concat(parts, ignore_index=True)


def add_sampling_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку weight - поправку на негативный сэмплинг.
    Считается ДО фильтра по TOP_N: для веса нужно знать, сколько случайных
    негативов взяли из группы всего, а не сколько их осталось в верхушке"""
    df = df.sort_values(["group_id", RANK_COL]).reset_index(drop=True)
    gid = df["group_id"]
    neg = df["label"] == 0

    # Порядковый номер негатива в группе по возрастанию RRF-ранга.
    # Трудные - первые не-позитивы RRF, поэтому это первые N_HARD
    # негативов группы: случайные брались из оставшихся и лежат ниже
    neg_order = pd.Series(-1, index=df.index)
    neg_order[neg] = df[neg].groupby("group_id").cumcount()
    is_hard = neg & (neg_order < N_HARD)
    is_rand = neg & ~is_hard

    # n_candidates — реальный размер списка кандидатов (n_candidates_total)
    assert (df.groupby("group_id")["n_candidates"].nunique() == 1).all(), \
        "n_candidates меняется внутри группы — признак считается не так"
    n_pos = df.groupby("group_id")["label"].transform("sum")
    n_hard = is_hard.groupby(gid).transform("sum")
    n_rand = is_rand.groupby(gid).transform("sum")

    # Из скольких негативов тянули случайные: все кандидаты минус позитивы
    # минус трудные. Вес = обратная доля взятых
    pool_neg = df["n_candidates"] - n_pos - n_hard
    w = np.ones(len(df))
    m = (is_rand & (n_rand > 0)).to_numpy()
    # clip(1): если негативов было мало, _sample брал всех — вес 1
    w[m] = (pool_neg[m] / n_rand[m]).clip(lower=1.0).to_numpy()
    df["weight"] = w

    print(f"трудных негативов: {is_hard.sum()}, случайных: {is_rand.sum()}, "
          f"средний вес случайного: {w[m].mean():.1f}")
    return df


def print_pos_rate_by_rank(df: pd.DataFrame, base: int) -> None:
    """Доля позитивов по рангам RRF: без весов и с весами.
    Без весов ожидается перекос (на рангах 20+ почти одни позитивы).
    С весами доля должна убывать с рангом"""
    bins = [0, 15, 30, 50, 100, 150, 500, 10**6]
    rb = pd.cut(df[RANK_COL] - base, bins, right=False)
    t = df.assign(rb=rb, wl=df["label"] * df["weight"]).groupby("rb", observed=True)
    rep = pd.DataFrame({
        "строк": t.size(),
        "доля_поз": t["label"].mean(),
        "доля_поз_взвеш": t["wl"].sum() / t["weight"].sum(),
    })
    print(rep.round(4))


def main():
    df = load_trainsets()
    print("до фильтра:", len(df), "строк,", df["group_id"].nunique(), "групп")

    # Ранг может считаться с 0 или с 1: берём минимум как базу
    base = int(df[RANK_COL].min())
    print(f"ранг считается с {base}")

    df = add_sampling_weights(df)
    print_pos_rate_by_rank(df, base)
    head = df[df[RANK_COL] < base + TOP_N]

    # Оставляем группы, где есть и позитив, и негатив:
    # группа из одного класса ничему не учит ранжированию
    g = head.groupby("group_id")["label"].agg(["sum", "count"])
    ok = g.index[(g["sum"] > 0) & (g["sum"] < g["count"])]
    head = head[head["group_id"].isin(ok)]
    print("после:", len(head), "строк,", head["group_id"].nunique(), "групп,",
          f"доля позитивов {head['label'].mean():.3f}")

    # Отложенные группы для early stopping
    rng = np.random.default_rng(SEED)
    groups = head["group_id"].unique()
    holdout = set(rng.choice(groups, size=len(groups) // 10, replace=False))
    is_ho = head["group_id"].isin(holdout)

    tr, ho = head[~is_ho], head[is_ho]
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=100,
        thread_count=-1,
        verbose=100,
        random_seed=SEED,
    )
    model.fit(
        Pool(tr[FEATURES], tr["label"], weight=tr["weight"]),
        eval_set=Pool(ho[FEATURES], ho["label"], weight=ho["weight"]),
    )
    model.save_model(str(OUT_MODEL))
    print(f"Модель сохранена: {OUT_MODEL}")
    print(model.get_feature_importance(prettified=True).head(10))


if __name__ == "__main__":
    main()