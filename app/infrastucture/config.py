# Все пути и общие константы проекта

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

TRAIN_FILE = DATA_RAW / "train.parquet"
BENCHMARK_Q_FILE = DATA_RAW / "benchmark_queries.parquet"
BENCHMARK_I_FILE = DATA_RAW / "benchmark_items.parquet"

ITEMS_EMBEDDINGS_FILE = DATA_PROCESSED / "items_embeddings.npz"
MICROCAT_MODEL_FILE = ROOT / "models" / "microcat_classifier.joblib"
ANSWER_FILE = ROOT / "answers.csv"
ALL_ITEMS_FILE = DATA_PROCESSED / "all_items.parquet"
E5_UNION_EMBEDDINGS_FILE = DATA_PROCESSED / "e5_union_embeddings.npz"
FRIDA_EMBEDDINGS_FILE = DATA_PROCESSED / "frida_union_embeddings.npz"
RERANKER_MODEL_FILE = ROOT / "models" / "catboost_reranker.cbm"
BM25_INDEX_FILE = DATA_PROCESSED / "bm25_index.joblib"

SEED = 67

# Поля search
# normed_search_query — нормализованный текст запроса (создаётся при загрузке).
SEARCH_COLS = ['normed_search_query', 'search_location_id',
               'search_is_delivery_search',
               'search_infm_params_text', 'search_category']
# Колонки-идентификаторы, для которых нужно превращение в строки
ID_COLS = ['item_id', 'query_id', 'search_location_id', 'item_location_id',
           'item_category_id', 'item_microcat_id', 'search_category']

