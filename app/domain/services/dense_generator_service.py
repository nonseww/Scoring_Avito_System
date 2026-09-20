from pathlib import Path
import numpy as np
from app.domain.services.embeddings_service import EmbeddingsService
from app.infrastucture.config import ITEMS_EMBEDDINGS_FILE, BENCHMARK_I_FILE
from app.infrastucture.loading import load_bench_items


class DenseGeneratorService:
    """Источник кандидатов на основе плотных векторов e5-small"""
    def __init__(self, embeddings_service: EmbeddingsService,
                 query_prefix: str = "query: ",
                 doc_prefix: str = "passage: "):
        self.embeddings_service = embeddings_service

        self.vectors = None
        self.ids = None
        self.row_by_id = None
        self.query_prefix = query_prefix  # e5: "query: ", FRIDA: "search_query: "
        self.doc_prefix = doc_prefix  # e5: "passage: ", FRIDA: "search_document: "

    def load_index(self, path: Path):
        """Загружает предпосчитанные эмбеддинги из файла"""
        self.vectors, self.ids = self.embeddings_service.load(path)
        self.row_by_id = {item_id: row for row, item_id in enumerate(self.ids)}
        return self

    def encode_queries(self, queries: list) -> np.ndarray:
        """Кодирует запросы"""
        return self.embeddings_service.embed_batch(queries, prefix=self.query_prefix, batch_size=128)

    def generate_with_scores(self, query_vector: np.ndarray, pool: list, top_k: int = 200) -> tuple:
        """Принимает закодированный вектор запроса.
        Возвращает (список item_id по убыванию близости, {item_id: скор})."""
        pool_rows = np.fromiter(
            (self.row_by_id[i] for i in pool if i in self.row_by_id),
            dtype=np.int64
        )
        if pool_rows.size == 0:
            return [], {}

        # Скалярное произведение = косинусная близость
        pool_vectors = self.vectors[pool_rows]
        pool_scores = pool_vectors @ query_vector

        if pool_scores.size > top_k:
            part = np.argpartition(-pool_scores, top_k)[:top_k]
        else:
            part = np.arange(pool_scores.size)

        order = part[np.argsort(-pool_scores[part])]

        ids = [self.ids[r] for r in pool_rows[order]]
        scores = {
            self.ids[r]: float(pool_scores[i])
            for i, r in zip(order, pool_rows[order])
        }
        return ids, scores

    def encode_documents(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Кодирует объявления. Используется скриптом построения индекса."""
        return self.embeddings_service.embed_batch(
            texts, prefix=self.doc_prefix, batch_size=batch_size
        )

    def build_index(self, all_items, path: Path, batch_size: int = 64):
        """Кодирует корпус и сохраняет матрицу. Разовая операция."""
        vectors = self.encode_documents(all_items["item_text"].tolist(), batch_size)
        self.embeddings_service.save(vectors, all_items["item_id"].tolist(), path)
        return self

    def generate(self, query_vector, pool, top_k=200) -> list:
        ids, _ = self.generate_with_scores(query_vector, pool, top_k)
        return ids


if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    print(">> Загрузка закончена")
    dense = DenseGeneratorService(EmbeddingsService()).load_index(ITEMS_EMBEDDINGS_FILE)
    print(f"vectors: {dense.vectors.shape}, ids: {len(dense.ids)}")
    print(f"норма первого вектора: {np.linalg.norm(dense.vectors[0]):.4f}")

    qv = dense.encode_queries(["наращивание ресниц"])[0]
    for item_id in dense.generate(qv, items["item_id"].tolist(), top_k=5):
        print(items.loc[items["item_id"] == item_id, "item_text"].iloc[0][:90])