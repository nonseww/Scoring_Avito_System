import time
from app.domain.services.bm25_generator_service import BM25GeneratorService
from app.domain.services.dense_generator_service import DenseGeneratorService
from app.domain.services.embeddings_service import EmbeddingsService
from app.domain.services.evaluation_service import EvaluationService
from app.domain.services.feature_extractor import FeatureExtractor
from app.domain.services.location_filter_service import LocationFilterService
from app.domain.services.microcat_generator_service import MicrocatGeneratorService
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.domain.services.rrf_service import RRFService
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE, ITEMS_EMBEDDINGS_FILE, \
    MICROCAT_MODEL_FILE, ANSWER_FILE, E5_UNION_EMBEDDINGS_FILE, FRIDA_EMBEDDINGS_FILE, ALL_ITEMS_FILE, BM25_INDEX_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train
from app.preprocessing.splitting import add_qid_in_train, split_train, make_eval_set
import pandas as pd
import numpy as np


def recall_no_cutoff(predictions: dict, gold: dict) -> float:
    """Получить предсказания без обрезки в 50 кандидатов
    (нужно для оценки потолка)"""
    query_recalls = []
    for qid, relevant in gold.items():
        pred_set = set(predictions.get(qid, []))
        gold_set = set(relevant)
        query_recalls.append(len(pred_set & gold_set) / len(gold_set))
    return sum(query_recalls) / len(query_recalls)


class Pipeline:
    def __init__(self):
        self.evaluation_service = EvaluationService()
        self.location_service = None
        self.popularity_service = None
        self.microcat_service = None
        self.bm25_service = None
        self.e5_service = None
        self.frida_service = None
        self.feature_extractor = None
        self.all_items = None
        self.bench_ids = None
        self.rrf_service = RRFService(k=60)

    def _setup(self):
        """Общая инициализация. Все индексы — по объединению (515 895).
        Ограничение на корпус бенчмарка делается пулом, а не индексом."""
        print(">> Загрузка данных")
        items = load_bench_items(BENCHMARK_I_FILE)
        train = load_train(TRAIN_FILE)
        self.all_items = pd.read_parquet(ALL_ITEMS_FILE)
        self.bench_ids = set(items["item_id"])

        print(">> Разбиение train")
        train = add_qid_in_train(train)
        train = split_train(train, benchmark_ids=self.bench_ids)

        print(">> Сервисы (индексы по объединению)")
        self.location_service = LocationFilterService(self.all_items, train)
        self.popularity_service = PopularityGeneratorService(self.all_items)
        self.microcat_service = MicrocatGeneratorService.load(
            MICROCAT_MODEL_FILE,
            popularity_service=self.popularity_service,
            top_k_classes=5
        )

        print(">> BM25 по объединению")
        t0 = time.time()
        if BM25_INDEX_FILE.exists():
            self.bm25_service = BM25GeneratorService.load(
                BM25_INDEX_FILE, popularity_service=self.popularity_service
            )
            print(f"загружен из кэша за {time.time() - t0:.0f} сек")
        else:
            self.bm25_service = BM25GeneratorService(
                popularity_service=self.popularity_service
            ).fit(self.all_items)
            self.bm25_service.save(BM25_INDEX_FILE)
            print(f"построен и сохранён за {time.time() - t0:.0f} сек")

        print(">> Dense-сервисы")
        self.e5_service = DenseGeneratorService(
            EmbeddingsService("intfloat/multilingual-e5-small"),
            query_prefix="query: ", doc_prefix="passage: "
        ).load_index(E5_UNION_EMBEDDINGS_FILE)
        self.frida_service = DenseGeneratorService(
            EmbeddingsService("ai-forever/FRIDA"),
            query_prefix="search_query: ", doc_prefix="search_document: "
        ).load_index(FRIDA_EMBEDDINGS_FILE)

        self.feature_extractor = FeatureExtractor(
            self.all_items, self.location_service, self.popularity_service
        )
        return train

    def _run_sources(self, query_text, qv_e5, qv_frida, microcats,
                     location_id, is_delivery, restrict_to, top_k=500):
        """Один запрос → кандидаты и всё, что нужно для признаков.
        Вызывается одинаково при обучении и на инференсе —
        различие только в restrict_to."""

        pool = self.location_service.get_pool(
            location_id, is_delivery=is_delivery, restrict_to=restrict_to
        )

        bm25_ids, bm25_scores = self.bm25_service.generate_with_scores(query_text, pool, top_k)
        e5_ids, e5_scores = self.e5_service.generate_with_scores(qv_e5, pool, top_k)
        frida_ids, frida_scores = self.frida_service.generate_with_scores(qv_frida, pool, top_k)
        microcat_ids = self.microcat_service.generate(query_text, pool, top_k, microcats=microcats)

        rankings = {
            "bm25": bm25_ids, "e5": e5_ids,
            "frida": frida_ids, "microcat": microcat_ids,
        }
        rrf_ids, rrf_scores = self.rrf_service.run_with_scores(
            rankings,
            weights={"bm25": 1.0, "e5": 0.75, "frida": 0.75, "microcat": 0.5},
            top_k=None  # для признаков нужен весь список, не топ-50
        )
        rankings["rrf"] = rrf_ids

        candidates = list(set(bm25_ids) | set(e5_ids) | set(frida_ids) | set(microcat_ids))

        return {
            "pool": pool,
            "candidates": candidates,
            "rankings": rankings,
            "scores": {"bm25": bm25_scores, "e5": e5_scores,
                       "frida": frida_scores, "rrf": rrf_scores},
        }

    def process(self):
        """Метод, нужный для обработки validation и получения по нему данных.
        Используется для оценки эффективности программы"""
        train = self._setup()
        val_queries, val_gold = make_eval_set(train, "val")
        gold_dict = dict(zip(val_gold["qid"], val_gold["item_ids"]))
        print(f"Запросов для оценки: {len(val_queries)}")

        texts = val_queries["normed_search_query"].tolist()

        print(">> Подкатегории и кодирование запросов")
        all_microcats = self.microcat_service.predict_microcats(texts)
        qvs_e5 = self.e5_service.encode_queries(texts)
        qvs_frida = self.frida_service.encode_queries(texts)

        print(">> Прогон по val")
        t0 = time.time()
        rankings_cached = {}
        union_no_cutoff = {}

        for i, row in enumerate(val_queries.itertuples()):
            out = self._run_sources(
                row.normed_search_query, qvs_e5[i], qvs_frida[i], all_microcats[i],
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1),
                restrict_to=self.bench_ids,
                top_k=500,
            )

            rankings_cached[row.qid] = out["rankings"]
            union_no_cutoff[row.qid] = out["candidates"]

        print(f"time = {time.time() - t0:.0f} секунд")

        print("\n>> Источники @50")

        for name in ["bm25", "e5", "frida", "microcat", "rrf"]:
            preds = {q: r[name][:50] for q, r in rankings_cached.items()}
            print(f"  {name:10} {self.evaluation_service.recall(preds, gold_dict):.4f}")

        print("\n>> Покрытие (без обрезки)")

        for name in ["bm25", "e5", "frida", "microcat"]:
            preds = {q: r[name] for q, r in rankings_cached.items()}
            print(f"  {name:10} {recall_no_cutoff(preds, gold_dict):.4f}")
        print(f"  union      {recall_no_cutoff(union_no_cutoff, gold_dict):.4f}")

        sizes = [len(v) for v in union_no_cutoff.values()]
        print(f"\nРазмер union: медиана {np.median(sizes):.0f}, среднее {np.mean(sizes):.0f}")

    def predict_answer(self):
        """Метод, предсказывающий итоговый ответ по бенчмарку"""
        self._setup()
        queries = load_bench_queries(BENCHMARK_Q_FILE)
        texts = queries["normed_search_query"].tolist()

        print(">> Подкатегории и кодирование запросов")
        all_microcats = self.microcat_service.predict_microcats(texts)
        qvs_e5 = self.e5_service.encode_queries(texts)
        qvs_frida = self.frida_service.encode_queries(texts)

        print(">> Прогон по бенчмарку")
        t0 = time.time()
        predictions = {}
        for i, row in enumerate(queries.itertuples()):
            out = self._run_sources(
                row.normed_search_query, qvs_e5[i], qvs_frida[i], all_microcats[i],
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1),
                restrict_to=self.bench_ids,
                top_k=500,
            )
            predictions[row.query_id] = out["rankings"]["rrf"][:50]
        print(f"time = {time.time() - t0:.0f} секунд")

        sizes = [len(v) for v in predictions.values()]
        print(f"Запросов: {len(predictions)} (ожидаем {len(queries)})")
        print(f"Кандидатов: медиана {np.median(sizes):.0f}, мин {min(sizes)}, макс {max(sizes)}")
        print(f"Пустых ответов: {sum(1 for s in sizes if s == 0)}")

        self.evaluation_service.save_answer(predictions, ANSWER_FILE)
        print(f">> Сохранено в {ANSWER_FILE}")
