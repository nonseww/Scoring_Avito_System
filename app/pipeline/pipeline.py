from app.domain.services.embeddings_service import EmbeddingsService
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE, ITEMS_EMBEDDINGS_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train


class Pipeline:
    def __init__(self):
        self.embeddings_service = EmbeddingsService()

    def process(self):
        print(">> Загрузка items")
        items = load_bench_items(BENCHMARK_I_FILE)
        print(">> Загрузка queries")
        # queries = load_bench_queries(BENCHMARK_Q_FILE)
        print(">> Загрузка train")
        # train = load_train(TRAIN_FILE)

        item_ids = items["item_id"].tolist()
        item_texts = items["item_text"].tolist()
        items_embeddings = self.embeddings_service.embed_batch(item_texts, prefix="passage: ", batch_size=128)
        self.embeddings_service.save(items_embeddings, item_ids, ITEMS_EMBEDDINGS_FILE)

