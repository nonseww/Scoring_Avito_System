import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import time
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.infrastucture.config import BENCHMARK_I_FILE, TRAIN_FILE
from app.infrastucture.loading import load_bench_items, load_train

TOKEN_RE = re.compile(r"[а-яa-z0-9]+")

class Lemmatizer:
    def __init__(self):
        self._morph = MorphAnalyzer()
        self._cache = {}

    def __call__(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text)
        lemmas = []
        for token in tokens:
            lemma = self._cache.get(token)
            if lemma is None:
                lemma = self._morph.parse(token)[0].normal_form
                self._cache[token] = lemma
            lemmas.append(lemma)
        return lemmas

class BM25GeneratorService:
    def __init__(self, k1: float= 1.5, b: float = 0.75, popularity_service: PopularityGeneratorService = None):
        self.k1 = k1  # насколько быстро насыщается вклад повторов слова
        self.b = b  # сила нормализации по длине документа (0 — выключена)
        self.popularity_service = popularity_service
        self.lemmatizer = Lemmatizer()

        self.vocab = None  # term -> номер колонки
        self.weights = None  # CSC-матрица (n_docs x n_terms) с предпосчитанными весами
        self.ids = None  # np.array item_id, порядок = строки матрицы
        self.row_by_id = None  # item_id -> номер строки

    def fit(self, items: pd.DataFrame):
        """Строит индекс по бенчмарку"""
        self.ids = items["item_id"].to_numpy()
        self.row_by_id = {item_id: row for row, item_id in enumerate(self.ids)}

        vectorizer = CountVectorizer(analyzer=self.lemmatizer, min_df=2)
        counts = vectorizer.fit_transform(items["item_text"].tolist())
        self.vocab = vectorizer.vocabulary_
        n_docs, n_terms = counts.shape
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        average_doc_len = doc_len.mean()
        df = np.bincount(counts.indices, minlength=n_terms)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        # --- предподсчёт весов ---
        # Ключевая идея: в формуле BM25
        #   score(D,Q) = Σ_t  idf[t] · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
        # всё, кроме самого факта «терм t есть в запросе», зависит только от пары (документ, терм).
        # Значит это можно посчитать заранее, а скоринг запроса свести к сложению колонок.

        # Для каждого ненулевого элемента матрицы нужно знать, в какой он строке.
        # В CSR это восстанавливается из indptr: строка i занимает indptr[i]:indptr[i+1].
        rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
        cols = counts.indices
        tf = counts.data.astype(np.float32)

        norm = 1 - self.b + self.b * doc_len[rows] / average_doc_len # нормировка по длине документа
        data = idf[cols] * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        self.weights = sparse.csc_matrix(
            (data.astype(np.float32), (rows, cols)), shape=(n_docs, n_terms)
        )

        return self

    def generate_with_scores(self, query_text: str, pool: list, top_k: int = 200) -> tuple:
        """Возвращает до top_k item_id из пула, упорядоченных по убыванию"""
        pool_rows = np.fromiter(
            (self.row_by_id[i] for i in pool if i in self.row_by_id),
            dtype=np.int64
        )
        if pool_rows.size == 0:
            return [], {}

        # Лемматизируем запрос тем эе анализатором
        lemmas = set(self.lemmatizer(query_text))
        term_ids = [self.vocab[t] for t in lemmas if t in self.vocab]

        if not term_ids:
            # Ничего нет - модель бессильна
            return self.popularity_service.generate_popularity_ranking(pool, top_k), {}

        # Сумма нужных колонок = вектор скоров по всем документам корпуса.
        # Колонки разрежены, так что работы тут ровно на те документы,
        # где эти термы реально встречаются.
        scores = np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()
        pool_scores = scores[pool_rows]

        # Документы с 0 не содержат ни одного терма запроса - не кандидаты
        nonzero = pool_scores > 0
        pool_rows = pool_rows[nonzero]
        pool_scores = pool_scores[nonzero]

        if pool_rows.size == 0:
            return self.popularity_service.generate_popularity_ranking(pool, top_k), {}

        # argpartition находит top_k без полной сортировки — O(n) вместо O(n log n).
        if pool_scores.size > top_k:
            part = np.argpartition(-pool_scores, top_k)[:top_k]
        else:
            part = np.arange(pool_scores.size)

        # Внутри отобранных - сортировка
        order = part[np.argsort(-pool_scores[part])]
        ids = [self.ids[r] for r in pool_rows[order]]
        scores = {self.ids[r]: float(s)
                  for r, s in zip(pool_rows[order], pool_scores[order])}
        return ids, scores

    def save(self, path: Path) -> None:
        sparse.save_npz(str(path.with_suffix(".npz")), self.weights)
        joblib.dump(
            {"vocab": self.vocab, "ids": self.ids, "k1": self.k1, "b": self.b},
            path
        )

    def generate(self, query_vector, pool, top_k=200) -> list:
        ids, _ = self.generate_with_scores(query_vector, pool, top_k)
        return ids

    @classmethod
    def load(cls, path: Path, popularity_service=None):
        data = joblib.load(path)
        service = cls(k1=data["k1"], b=data["b"], popularity_service=popularity_service)
        service.vocab = data["vocab"]
        service.ids = data["ids"]
        service.row_by_id = {item_id: row for row, item_id in enumerate(service.ids)}
        service.weights = sparse.load_npz(str(path.with_suffix(".npz")))
        return service

if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    print(">> Закончили загрузку данных")
    popul_service = PopularityGeneratorService(items)
    bm25 = BM25GeneratorService(popularity_service=popul_service)
    t0 = time.time()
    bm25.fit(items)
    print(f"fit занял {time.time() - t0:.0f} сек")

    print(f"размер словаря: {len(bm25.vocab)}")
    print(f"размер матрицы: {bm25.weights.shape}, ненулевых: {bm25.weights.nnz:,}")
    print(f"уникальных лемм в кэше: {len(bm25.lemmatizer._cache):,}")

    for q in ["шугаринг глубокое бикини", "ремонт стиральных машин", "маникюр гель лак"]:
        result = bm25.generate(q, items["item_id"].tolist(), top_k=5)
        print(f"\nЗапрос: {q}")
        for item_id in result:
            text = items.loc[items["item_id"] == item_id, "item_text"].iloc[0]
            print(f"   {text[:90]}")