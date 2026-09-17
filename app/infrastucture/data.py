import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from app.infrastucture.config import ID_COLS, BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE
from app.preprocessing.normalize import build_item_text, normalize_text


def read_parquet_clean(path, columns=None) -> pd.DataFrame:
    """Читает parquet-файл и переводит decimal128 в float64,
    колонки-идентификаторы к строкам"""
    table = pq.read_table(path, columns=columns)
    decimal_cols = [f.name for f in table.schema if pa.types.is_decimal(f.type)]
    df = table.to_pandas()
    for col in decimal_cols:
        df[col] = df[col].astype('float64')
    for col in ID_COLS:
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_float_dtype(s):
                s = s.astype("Int64")
            df[col] = s.astype(str)
    return df

def load_bench_items(path) -> pd.DataFrame:
    """Загрузка корпуса объявлений, где ищем кандидатов"""
    items = read_parquet_clean(path)

    assert items["item_id"].is_unique, "в бенчмарке есть повторяющиеся item_id"
    assert items["item_id"].str.fullmatch(r'[0-9a-f]{16}').all(), "неверный формат item_id"

    items["item_text"] = build_item_text(items)
    items = items.drop(columns=["item_description_raw"])
    return items

def load_bench_queries(path) -> pd.DataFrame:
    """Загрузка запросов бенчмарка, для которых нужен ответ"""
    queries = read_parquet_clean(path)
    assert queries["query_id"].is_unique, "в бенчмарке есть повторяющиеся query_id"
    queries["normed_search_query"] = normalize_text(queries["search_query"])
    queries["search_infm_params_text"] = queries["search_infm_params_text"].fillna('')
    queries["qid"] = queries["query_id"]
    return queries

def load_train(path) -> pd.DataFrame:
    """Загрузка train данных"""
    train = read_parquet_clean(path)
    train["normed_search_query"] = normalize_text(train["search_query"])
    train["search_infm_params_text"] = train["search_infm_params_text"].fillna('')
    train["item_text"] = build_item_text(train)
    train = train.drop(columns=["item_description_raw"])
    return train

if __name__ == "__main__":
    items = load_bench_items(BENCHMARK_I_FILE)
    queries = load_bench_queries(BENCHMARK_Q_FILE)
    train = load_train(TRAIN_FILE)

    print(items.shape, queries.shape, train.shape)
    print(items[["item_id", "item_text"]].head(2))
    print(queries[["quid", "normed_search_query"]].head(2))