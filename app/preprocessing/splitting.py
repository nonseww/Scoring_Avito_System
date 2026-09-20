import pandas as pd
import numpy as np
from app.infrastucture.config import SEARCH_COLS, SEED, BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train


def add_qid_in_train(train: pd.DataFrame) -> pd.DataFrame:
    """Добавление столбца qid в train"""
    group_numbers = train.groupby(SEARCH_COLS, sort=False, dropna=False).ngroup()
    train["qid"] = "tr_" + group_numbers.astype(str)
    return train

def split_train(train: pd.DataFrame, benchmark_ids: set,
                n_val: int = 3000, n_rank: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Деление train на fit (статистики и индексы), rank (обучение перереранкера),
    val (локальная оценка)"""

    # У каких qid все объявления есть в бенчмарке
    in_benchmark = train["item_id"].isin(benchmark_ids)
    eligible_by_qid = in_benchmark.groupby(train["qid"]).all()
    train["eligible"] = train["qid"].map(eligible_by_qid)

    # номер группы "текст + локация"
    train["grp"] = train.groupby(
        ["normed_search_query", "search_location_id"], sort=False
    ).ngroup()

    # сколько eligible-запросов в каждой группе?
    qids_per_grp = (
        train.loc[train["eligible"]]
        .groupby("grp")["qid"]
        .nunique()
    )

    # перемешиваем группы и выбирает val/rank
    rand_groups = qids_per_grp.sample(frac=1, random_state=SEED)
    cum = rand_groups.cumsum()
    val_groups = cum.index[cum <= n_val]
    rank_groups = cum.index[(cum > n_val) & (cum <= n_val + n_rank)]

    # запись результата в split
    train["split"] = "fit"
    train.loc[train["grp"].isin(rank_groups), "split"] = "rank"
    train.loc[train["grp"].isin(val_groups), "split"] = "val"
    return train

def make_eval_set(train: pd.DataFrame, split: str, eligible_only: bool = True):
    """Для val/rank возвращает:
    queries - одна строка на запрос
    gold - для каждого qid список правильных item_id

    eligible_only=True - только запросы, чьи ответы есть в корпусе бенчмарка
    (нужно для честной оценки: недостижимый ответ занижал бы recall).
    eligible_only=False - все запросы сплита; для обучения реранкера,
    где кандидаты берутся из объединения и ответ из train достижим"""
    part = train[train["split"] == split]
    if eligible_only:
        part = part[part["eligible"]]

    queries = (
        part.drop_duplicates("qid")[["qid"] + SEARCH_COLS + ["search_query"]]
        .reset_index(drop=True)
    )

    gold = (
        part.groupby("qid")['item_id']
        .apply(lambda s: sorted(set(s)))
        .rename("item_ids")
        .reset_index()
    )
    return queries, gold

if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    print(">> Загрузка queries")
    queries = load_bench_queries(BENCHMARK_Q_FILE)
    print(">> Загрузка train")
    train = load_train(TRAIN_FILE)

    train = add_qid_in_train(train)
    train = split_train(train, benchmark_ids=set(items["item_id"]))
    val_queries, val_gold = make_eval_set(train, "val")
    rank_queries, rank_gold = make_eval_set(train, "rank")

    print("Строк по каждой части:")
    print(train["split"].value_counts())
    print(f"val для оценки: {len(val_queries)}")
    print(f"rank для оценки: {len(rank_queries)}")

    fit_part = train[train["split"] == "fit"]
    fit_texts = set(fit_part["normed_search_query"])
    fit_text_loc = set(zip(fit_part["normed_search_query"], fit_part["search_location_id"]))
    seen_text = val_queries["normed_search_query"].isin(fit_texts).mean()
    seen_text_loc = [
        (q, loc) in fit_text_loc
        for q, loc in zip(val_queries["normed_search_query"], val_queries["search_location_id"])
    ]
    print(f"val: текст знаком fit {seen_text:.1%}")
    print(f"val: текст+локация знакомы fit {sum(seen_text_loc) / len(seen_text_loc):.1%} (в бенчмарке было 6.6%)")