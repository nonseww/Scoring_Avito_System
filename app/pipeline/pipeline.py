import time
from app.domain.services.embeddings_service import EmbeddingsService
from app.domain.services.evaluation_service import EvaluationService
from app.domain.services.location_filter_service import LocationFilterService
from app.domain.services.microcat_generator_service import MicrocatGeneratorService
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE, ITEMS_EMBEDDINGS_FILE, \
    MICROCAT_MODEL_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train
from app.preprocessing.splitting import add_qid_in_train, split_train, make_eval_set
import numpy as np

def recall_no_cutoff(predictions: dict, gold: dict) -> float:
    query_recalls = []
    for qid, relevant in gold.items():
        pred_set = set(predictions.get(qid, []))
        gold_set = set(relevant)
        query_recalls.append(len(pred_set & gold_set) / len(gold_set))
    return sum(query_recalls) / len(query_recalls)

class Pipeline:
    def __init__(self):
        # self.embeddings_service = EmbeddingsService()
        self.evaluation_service = EvaluationService()
        self.location_service = None
        self.popularity_service = None
        self.microcat_service = None

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
        self.microcat_service = MicrocatGeneratorService.load(MICROCAT_MODEL_FILE, popularity_service=self.popularity_service)

        print(">> Предсказание подкатегорий для всех запросов сразу")
        all_microcats = self.microcat_service.predict_microcats(
            val_queries["normed_search_query"].tolist()
        )

        print(">> Прогон по val-запросам")
        t0 = time.time()
        predictions = {}
        for i, row in enumerate(val_queries.itertuples()):
            pool = self.location_service.get_pool(
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1)
            )
            predictions[row.qid] = self.microcat_service.generate(
                row.normed_search_query, pool, top_k=50, microcats=all_microcats[i]
            )
        print(f"time = {time.time() - t0:.0f} секунд")

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
        sizes = []
        candidates_no_cutoff = {}
        for i, row in enumerate(val_queries.itertuples()):
            pool = self.location_service.get_pool(
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1)
            )
            candidate_ids = set()
            for cat in all_microcats[i]:
                candidate_ids.update(self.microcat_service.items_by_microcat.get(cat, []))
            pool_after_microcat = [x for x in pool if x in candidate_ids]
            sizes.append(len(pool_after_microcat))
            candidates_no_cutoff[row.qid] = pool_after_microcat

        print(f"после сужения по подкатегориям: медиана {np.median(sizes):.0f}, среднее {np.mean(sizes):.0f}")
        print(f"запросов, где осталось <=50 кандидатов: {np.mean(np.array(sizes) <= 50):.1%}")

        ceiling_recall = recall_no_cutoff(candidates_no_cutoff, gold_dict)
        print(f"Потолок Recall (без обрезки до 50) после сужения по подкатегориям: {ceiling_recall:.4f}")