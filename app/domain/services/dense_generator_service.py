from pathlib import Path
import numpy as np
from app.domain.services.embeddings_service import EmbeddingsService
from app.infrastucture.config import ITEMS_EMBEDDINGS_FILE, BENCHMARK_I_FILE
from app.infrastucture.loading import load_bench_items


class DenseGeneratorService:
    """Источник кандидатов на основе плотных векторов e5-small"""
    def __init__(self, embeddings_service: EmbeddingsService):
        self.embeddings_service = embeddings_service

        self.vectors = None
        self.ids = None
        self.row_by_id = None

    def load_index(self, path: Path):
        """Загружает предпосчитанные эмбеддинги из файла"""
        self.vectors, self.ids = self.embeddings_service.load(path)
        self.row_by_id = {item_id: row for row, item_id in enumerate(self.ids)}
        return self

    def encode_queries(self, queries: list) -> np.ndarray:
        """Кодирует запросы"""
        return self.embeddings_service.embed_batch(queries, prefix="query: ", batch_size=128)

    def generate(self, query_vector:np.ndarray, pool: list, top_k: int = 200) -> list:
        """Принимает закодированный вектор запроса"""
        pool_rows = np.fromiter(
            (self.row_by_id[i] for i in pool if i in self.row_by_id),
            dtype=np.int64
        )
        if pool_rows.size == 0:
            return []

        # Скалярное произведение = косинусная близость
        pool_vectors = self.vectors[pool_rows]
        scores = pool_vectors @ query_vector

        if scores.size > top_k:
            part = np.argpartition(-scores, top_k)[:top_k]
        else:
            part = np.arange(scores.size)

        order = part[np.argsort(-scores[part])]
        return [self.ids[r] for r in pool_rows[order]]

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