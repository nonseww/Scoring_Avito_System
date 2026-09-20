import pandas as pd
from app.infrastucture.config import BENCHMARK_I_FILE, TRAIN_FILE, ALL_ITEMS_FILE
from app.infrastucture.loading import load_bench_items, load_train

'''объединение объявлений бенчмарка и уникальных объявлений train 
по фиксированному набору колонок, дедупликация по item_id, 
два выхода - полный (all_items.parquet, с географией и 
рейтингом для feature_extractor) и облегчённый только 
с текстом (all_items_for_encoding.parquet, для BM25 и энкодеров)'''

items = load_bench_items(BENCHMARK_I_FILE)
train = load_train(TRAIN_FILE)

train_items = train.drop_duplicates("item_id")

cols = ["item_id", "item_text", "item_microcat_id", "item_rating",
        "item_rating_reviews_count", "item_latitude", "item_longitude",
        "item_location_id"]

all_items = pd.concat([items[cols], train_items[cols]]).drop_duplicates("item_id")
all_items = all_items.reset_index(drop=True)

print(f"Объединение: {len(all_items)}")        # 515 895
all_items.to_parquet(ALL_ITEMS_FILE, index=False)

all_items[["item_id", "item_text"]].to_parquet("all_items_for_encoding.parquet", index=False)