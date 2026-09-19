import time
from app.domain.services.bm25_generator_service import BM25GeneratorService
from app.domain.services.dense_generator_service import DenseGeneratorService
from app.domain.services.embeddings_service import EmbeddingsService
from app.domain.services.evaluation_service import EvaluationService
from app.domain.services.location_filter_service import LocationFilterService
from app.domain.services.microcat_generator_service import MicrocatGeneratorService
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.domain.services.rrf_service import RRFService
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE, ITEMS_EMBEDDINGS_FILE, \
    MICROCAT_MODEL_FILE, ANSWER_FILE
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
        self.embeddings_service = EmbeddingsService()
        self.evaluation_service = EvaluationService()
        self.location_service = None
        self.popularity_service = None
        self.microcat_service = None
        self.bm25_service = None
        self.dense_service = DenseGeneratorService(self.embeddings_service).load_index(ITEMS_EMBEDDINGS_FILE)
        self.rrf_service = RRFService()

    def predict_answer(self):
        print(">> Загрузка items")
        items = load_bench_items(BENCHMARK_I_FILE)
        # print(">> Загрузка queries")
        # queries = load_bench_queries(BENCHMARK_Q_FILE)
        # print(">> Загрузка train")
        train = load_train(TRAIN_FILE)
        print(">> Закончили загрузку данных")

        import pandas as pd
        ans = pd.read_csv("answers.csv")
        first_ids = ans.iloc[0]["answer"].split()
        print(set(first_ids) <= set(items["item_id"]))  # должно быть True
        print(ans["answer"].str.split().apply(len).describe())  # все ли по 50
        dups = ans["answer"].apply(lambda s: len(s.split()) != len(set(s.split())))
        print(f"Ответов с повторами: {dups.sum()}")

        train_items = train.drop_duplicates("item_id")
        all_ids = set(train_items["item_id"]) | set(items["item_id"])
        print(f"Уникальных в train: {len(train_items)}")
        print(f"Объединение: {len(all_ids)}")

        '''
        print(">> Инициализация сервисов")
        self.location_service = LocationFilterService(items, train)
        self.popularity_service = PopularityGeneratorService(items)
        self.microcat_service = MicrocatGeneratorService.load(
            MICROCAT_MODEL_FILE,
            popularity_service=self.popularity_service,
            top_k_classes=5
        )
        self.bm25_service = BM25GeneratorService(popularity_service=self.popularity_service).fit(items)

        print(">> Предсказание подкатегорий")
        all_microcats = self.microcat_service.predict_microcats(
            queries["normed_search_query"].tolist()
        )

        print(">> Кодирование запросов")
        qvs = self.dense_service.encode_queries(queries=queries["normed_search_query"].tolist())

        print(">> Прогон по бенчмарку")
        t0 = time.time()
        predictions = {}
        for i, row in enumerate(queries.itertuples()):
            location_pool = self.location_service.get_pool(
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1)
            )
            microcat_prediction = self.microcat_service.generate(
                row.normed_search_query, location_pool, top_k=500, microcats=all_microcats[i]
            )
            bm25_prediction = self.bm25_service.generate(
                row.normed_search_query, location_pool, top_k=500
            )
            dense_prediction = self.dense_service.generate(
                qvs[i], location_pool, top_k=500
            )
            predictions[row.query_id] = self.rrf_service.run(
                {
                    "bm25": bm25_prediction,
                    "microcat": microcat_prediction,
                    "dense": dense_prediction
                },
                weights={"bm25": 1.0, "microcat": 0.5, "dense": 0.75},
                top_k=50
            )
        print(f"time = {time.time() - t0:.0f} секунд")

        print(f"Запросов в бенчмарке: {len(queries)}")
        print(f"Запросов в ответе: {len(predictions)}")

        sizes = [len(v) for v in predictions.values()]
        print(f"Кандидатов: медиана {np.median(sizes):.0f}, минимум {min(sizes)}, максимум {max(sizes)}")
        print(f"Запросов с пустым ответом: {sum(1 for s in sizes if s == 0)}")

        self.evaluation_service.save_answer(predictions, ANSWER_FILE)
        print(f">> Сохранено в {ANSWER_FILE}")
        '''

    def process(self):
        print(">> Загрузка items")
        items = load_bench_items(BENCHMARK_I_FILE)
        print(">> Загрузка queries")
        queries = load_bench_queries(BENCHMARK_Q_FILE)
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

        # Кодируем queries в embeddings
        qvs = self.dense_service.encode_queries(queries=val_queries["normed_search_query"].tolist())
        print(">> Прогон по val-запросам")
        t0 = time.time()
        rrf_predictions = {}
        bm25_only = {}
        dense_only = {}
        microcat_only = {}
        source_cached = {}
        no_cutoff = {}
        for i, row in enumerate(val_queries.itertuples()):
            location_pool = self.location_service.get_pool(
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1)
            )
            microcat_prediction = self.microcat_service.generate(
                row.normed_search_query,
                location_pool,
                top_k=500,
                microcats=all_microcats[i]
            )
            bm25_prediction = self.bm25_service.generate(
                row.normed_search_query,
                location_pool,
                top_k=500
            )
            dense_prediction = self.dense_service.generate(qvs[i], location_pool, top_k=500)
            no_cutoff[row.qid] = list(set(bm25_prediction) | set(microcat_prediction) | set(dense_prediction))
            rrf_predictions[row.qid] = self.rrf_service.run(
                {
                    "bm25": bm25_prediction,
                    "microcat": microcat_prediction,
                    "dense": dense_prediction
                },
                weights={"bm25": 1.0, "microcat": 0.5, "dense": 1.0},
                top_k=50
            )
            source_cached[row.qid] = {
                "bm25": bm25_prediction,
                "microcat": microcat_prediction,
                "dense": dense_prediction
            }
            bm25_only[row.qid] = bm25_prediction[:50]
            microcat_only[row.qid] = microcat_prediction[:50]
            dense_only[row.qid] = dense_prediction[:50]

        print(f"time = {time.time() - t0:.0f} секунд")

        gold_dict = dict(zip(val_gold["qid"], val_gold["item_ids"]))

        print(f"BM25 один: {self.evaluation_service.recall(bm25_only, gold_dict):.4f}")
        print(f"Подкатегории: {self.evaluation_service.recall(microcat_only, gold_dict):.4f}")
        print(f"Dense: {self.evaluation_service.recall(dense_only, gold_dict):.4f}")
        print(f"RRF (1.0 / 1.0): {self.evaluation_service.recall(rrf_predictions, gold_dict):.4f}")

        print("\n Покрытие кандидатов (recall без обрезки до 50)")
        # по каждому источнику отдельно — где чей вклад
        print(f"BM25 (300): {recall_no_cutoff({q: s['bm25'] for q, s in source_cached.items()}, gold_dict):.4f}")
        print(
            f"Подкатегории: {recall_no_cutoff({q: s['microcat'] for q, s in source_cached.items()}, gold_dict):.4f}")
        print(f"Dense (300): {recall_no_cutoff({q: s['dense'] for q, s in source_cached.items()}, gold_dict):.4f}")
        print(f"Union всех трёх: {recall_no_cutoff(no_cutoff, gold_dict):.4f}")

        sizes = [len(v) for v in no_cutoff.values()]
        print(f"Размер union: медиана {np.median(sizes):.0f}, среднее {np.mean(sizes):.0f}")

        # и потолок пула локации для сравнения
        print(f"\nИтог RRF@50:      0.8102")
        print(f"Потолок локации:  0.988")

        # поиск оптимального веса для подкатегорий в RRF
        # for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        #     preds = {
        #         qid: self.rrf_service.run(src, weights={"bm25": 1.0, "microcat": w}, top_k=50)
        #         for qid, src in source_cached.items()
        #     }
        #     print(f"w_microcat = {w}: {self.evaluation_service.recall(preds, gold_dict):.4f}")
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            preds = {
                qid: self.rrf_service.run(src, weights={"bm25": 1.0, "microcat": 0.5, "dense": w})
                for qid, src in source_cached.items()
            }
            print(f"w_dense = {w}: {self.evaluation_service.recall(preds, gold_dict):.4f}")

        for k in [10, 20, 40, 60, 100]:
            rrf = RRFService(k=k)
            preds = {qid: rrf.run(src, weights={"bm25": 1.0, "microcat": 0.5, "dense": 0.75}, top_k=50)
                     for qid, src in source_cached.items()}
            print(f"k={k}: {self.evaluation_service.recall(preds, gold_dict):.4f}")
        # Построение эмбеддингов
        # item_ids = items["item_id"].tolist()
        # item_texts = items["item_text"].tolist()
        # items_embeddings = self.embeddings_service.embed_batch(item_texts, prefix="passage: ", batch_size=128)
        # self.embeddings_service.save(items_embeddings, item_ids, ITEMS_EMBEDDINGS_FILE)

