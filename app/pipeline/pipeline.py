import time

from pip._internal.models import candidate

from app.domain.services.bm25_generator_service import BM25GeneratorService
from app.domain.services.embeddings_service import EmbeddingsService
from app.domain.services.evaluation_service import EvaluationService
from app.domain.services.location_filter_service import LocationFilterService
from app.domain.services.microcat_generator_service import MicrocatGeneratorService
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.domain.services.rrf_service import RRFService
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
        self.bm25_service = None
        self.rrf_service = RRFService()

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
        self.microcat_service = MicrocatGeneratorService.load(
            MICROCAT_MODEL_FILE,
            popularity_service=self.popularity_service,
            top_k_classes=5
        )
        self.bm25_service = BM25GeneratorService(popularity_service=self.popularity_service).fit(items)

        print(">> Предсказание подкатегорий для всех запросов сразу")
        all_microcats = self.microcat_service.predict_microcats(
            val_queries["normed_search_query"].tolist()
        )

        # Пулы считаются один раз: локация + сужение по топ-5 подкатегориям.
        # Дальше их переиспользуют baseline, BM25 и в будущем dense/RRF —
        # так все источники честно сравниваются на одинаковых кандидатах.
        print(">> Прогон по val-запросам")
        t0 = time.time()
        rrf_predictions = {}
        bm25_only = {}
        microcat_only = {}
        source_cached = {}
        for i, row in enumerate(val_queries.itertuples()):
            location_pool = self.location_service.get_pool(
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1)
            )
            microcat_prediction = self.microcat_service.generate(
                row.normed_search_query,
                location_pool,
                top_k=300,
                microcats=all_microcats[i]
            )
            bm25_prediction = self.bm25_service.generate(
                row.normed_search_query,
                location_pool,
                top_k=300
            )
            rrf_predictions[row.qid] = self.rrf_service.run(
                {"bm25": bm25_prediction, "microcat": microcat_prediction},
                weights={"bm25": 1.0, "microcat": 0.5},
                top_k=50
            )
            source_cached[row.qid] = {"bm25": bm25_prediction, "microcat": microcat_prediction}
            bm25_only[row.qid] = bm25_prediction[:50]
            microcat_only[row.qid] = microcat_prediction[:50]

        print(f"time = {time.time() - t0:.0f} секунд")

        gold_dict = dict(zip(val_gold["qid"], val_gold["item_ids"]))

        print(f"BM25 один: {self.evaluation_service.recall(bm25_only, gold_dict):.4f}")
        print(f"Подкатегории: {self.evaluation_service.recall(microcat_only, gold_dict):.4f}")
        print(f"RRF (1.0 / 1.0): {self.evaluation_service.recall(rrf_predictions, gold_dict):.4f}")

        # поиск оптимального веса для подкатегорий в RRF
        # for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        #     preds = {
        #         qid: self.rrf_service.run(src, weights={"bm25": 1.0, "microcat": w}, top_k=50)
        #         for qid, src in source_cached.items()
        #     }
        #     print(f"w_microcat = {w}: {self.evaluation_service.recall(preds, gold_dict):.4f}")


        # Построение эмбеддингов
        # item_ids = items["item_id"].tolist()
        # item_texts = items["item_text"].tolist()
        # items_embeddings = self.embeddings_service.embed_batch(item_texts, prefix="passage: ", batch_size=128)
        # self.embeddings_service.save(items_embeddings, item_ids, ITEMS_EMBEDDINGS_FILE)

