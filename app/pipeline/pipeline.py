from app.domain.services.embeddings_service import EmbeddingsService
from app.domain.services.evaluation_service import EvaluationService
from app.domain.services.location_filter_service import LocationFilterService
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE, ITEMS_EMBEDDINGS_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train
from app.preprocessing.splitting import add_qid_in_train, split_train, make_eval_set
import numpy as np


class Pipeline:
    def __init__(self):
        # self.embeddings_service = EmbeddingsService()
        self.evaluation_service = EvaluationService()
        self.location_service = None
        self.popularity_service = None

    def process(self):
        print(">> Загрузка items")
        items = load_bench_items(BENCHMARK_I_FILE)
        # print(">> Загрузка queries")
        # queries = load_bench_queries(BENCHMARK_Q_FILE)
        print(">> Загрузка train")
        train = load_train(TRAIN_FILE)
        print(">> Закончили загрузку данных")

        print(">> Разбиение train")
        train = add_qid_in_train(train)
        train = split_train(train, benchmark_ids=set(items["item_id"]))
        val_queries, val_gold = make_eval_set(train, "val")
        print(f"Запросов для оценки: {len(val_queries)}")

        print(">> Фильтры по локации и рейтингу")
        self.location_service = LocationFilterService(items, train)
        self.popularity_service = PopularityGeneratorService(items)

        print(">> Прогон по val-запросам")
        predictions = {}
        for _, row in val_queries.iterrows():
            pool = self.location_service.get_pool(
                row["search_location_id"],
                is_delivery=(row["search_is_delivery_search"] == 1)
            )
            predictions[row["qid"]] = self.popularity_service.generate_popularity_ranking(pool, top_k=50)

        print(">> Оценка")
        gold_dict = dict(zip(val_gold["qid"], val_gold["item_ids"]))
        recall = self.evaluation_service.recall(predictions, gold_dict)
        print(f"Baseline Recall@50: {recall:.4f}")
        # Построение эмбеддингов
        # item_ids = items["item_id"].tolist()
        # item_texts = items["item_text"].tolist()
        # items_embeddings = self.embeddings_service.embed_batch(item_texts, prefix="passage: ", batch_size=128)
        # self.embeddings_service.save(items_embeddings, item_ids, ITEMS_EMBEDDINGS_FILE)

        print(">> Проверка покрытия пула")
        pool_hits = []
        pool_sizes = []
        for _, row in val_queries.iterrows():
            pool = self.location_service.get_pool(
                row["search_location_id"],
                is_delivery=(row["search_is_delivery_search"] == 1)
            )
            pool_set = set(pool)
            relevant = set(gold_dict.get(row["qid"], []))
            pool_hits.append(len(pool_set & relevant) / len(relevant))
            pool_sizes.append(len(pool))

        print(f"Recall пула (потолок): {np.mean(pool_hits):.4f}")
        print(f"размер пула: медиана {np.median(pool_sizes):.0f}, среднее {np.mean(pool_sizes):.0f}")
        median_pool = np.median(pool_sizes)
        print(f"ожидаемый recall при случайном выборе 50 из пула: {50 / median_pool:.4f}")