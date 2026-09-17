# Все пути и общие константы проекта

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

TRAIN_FILE = DATA_RAW / "train.parquet"
BENCHMARK_Q_FILE = DATA_RAW / "benchmark_queries.parquet"
BENCHMARK_FILE = DATA_RAW / "benchmark_items.parquet"

SEED = 67

# Поля search
# normed_query_search — нормализованный текст запроса (создаётся при загрузке).
SEARCH_COLS = ['normed_query_search', 'search_location_id',
               'search_is_delivery_search',
               'search_infm_params_text', 'search_category']
# Колонки-идентификаторы, для которых нужно превращение в строки
ID_COLS = ['item_id', 'query_id', 'search_location_id', 'item_location_id',
           'item_category_id', 'item_microcat_id', 'search_category']

