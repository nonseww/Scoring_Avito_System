from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

from app.infrastucture.config import BENCHMARK_I_FILE
from app.infrastucture.loading import load_bench_items


class EmbeddingsService:
    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model, device="cuda")

    def embed_batch(self, texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
        with_prefixes = [f"{prefix}{t}" for t in texts]
        return self.model.encode(
            with_prefixes,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed(self, text: str, prefix: str) -> np.ndarray:
        return self.embed_batch([text], prefix=prefix)[0]

    def save(self, vectors: np.ndarray, ids: list[str], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=vectors, ids=np.array(ids))

    def load(self, path: Path) -> tuple[np.ndarray, list[str]]:
        data = np.load(path, allow_pickle=True)
        return data["vectors"], data["ids"].tolist()


if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    sample_items = items.head(200)
    item_ids = sample_items["item_id"].tolist()
    item_texts = sample_items["item_text"].tolist()

    service = EmbeddingsService()

    vectors = service.embed_batch(item_texts, prefix="passage: ")
    print(f"Vector's shape: {vectors.shape}")
    assert vectors.shape[0] == len(item_ids), "Число векторов не совпало с количеством id"

    norms = np.linalg.norm(vectors, axis=1)
    print(f"Norms: {norms[:5]}")
    assert np.allclose(norms, 1.0, atol=1e-3), "Векторы не нормализованы"

    test_path = Path("data/embeddings/_test.npz")
    service.save(vectors, item_ids, test_path)
    print(f"Файл создан: {test_path.exists()}, размер: {test_path.stat().st_size} байт")

    loaded_vectors, loaded_ids = service.load(test_path)
    assert loaded_ids == item_ids, "id рассинхронились"
    assert np.allclose(loaded_vectors, vectors), "Векторы разъехались"
    print("Сохранения и загрузка корректны")

    query_vector = service.embed("автоподбор", prefix="query: ")
    sims = loaded_vectors @ query_vector
    top_idx = np.argsort(-sims)[:3]
    print("Топ 3 похожих")
    for i in top_idx:
        print(f"{sims[i]:.3f} {item_texts[i][:80]}")

    test_path.unlink()