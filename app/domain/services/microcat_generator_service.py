from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from app.domain.services.popularity_generator_service import PopularityGeneratorService
from app.infrastucture.config import BENCHMARK_I_FILE, TRAIN_FILE, MICROCAT_MODEL_FILE
from app.infrastucture.loading import load_bench_items, load_train
from app.preprocessing.splitting import add_qid_in_train, split_train, make_eval_set


class MicrocatGeneratorService:
    def __init__(self, popularity_service: PopularityGeneratorService, top_k_classes: int = 5):
        """Модель, предсказывающая подкатегорию запроса"""
        self.top_k_classes = top_k_classes
        self.popularity_service = popularity_service
        self.vectorizer = None
        self.classifier = None
        self.items_by_microcat = None

    def fit(self, fit_data: pd.DataFrame, items: pd.DataFrame):
        """Обучает классификатор на fit-части train и готовит структуру
        для быстрого отбора"""
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=200_000
        )
        X = self.vectorizer.fit_transform(fit_data["normed_search_query"])
        y = fit_data["item_microcat_id"]
        self.classifier = SGDClassifier(
            loss="log_loss",
            max_iter=15,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
            verbose=1
        )
        self.classifier.fit(X, y)

        self.items_by_microcat = items.groupby("item_microcat_id")["item_id"].apply(list).to_dict()
        return self

    def predict_microcats(self, queries: list) -> np.ndarray:
        X = self.vectorizer.transform(queries)
        proba = self.classifier.predict_proba(X)
        top_idx = np.argsort(proba, axis=1)[:, -self.top_k_classes:][:, ::-1]
        return self.classifier.classes_[top_idx]

    def generate(self, query_text: str, pool: list, top_k: int = 200, microcats=None):
        """если передам microcats, модель не вызывается повторно. Нужно для
        пакетного прогона"""

        if microcats is None:
            microcats = self.predict_microcats([query_text])[0]

        candidate_ids = set()
        for cat in microcats:
            candidate_ids.update(self.items_by_microcat.get(cat, []))
        filtered = [item_id for item_id in pool if item_id in candidate_ids]
        return self.popularity_service.generate_popularity_ranking(filtered, top_k=top_k)

    def save(self, path: Path) -> None:
        """Сохранение на диск"""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "items_by_microcat": self.items_by_microcat,
                "top_k_classes": self.top_k_classes
            },
            path,
            compress=3
        )

    @classmethod
    def load(cls, path: Path, popularity_service: PopularityGeneratorService, top_k_classes: int = 5):
        """Загружает ранее обученную модель"""
        data = joblib.load(path)
        service = cls(popularity_service, top_k_classes=top_k_classes)
        service.vectorizer = data["vectorizer"]
        service.classifier = data["classifier"]
        service.items_by_microcat = data["items_by_microcat"]
        return service


if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    print(">> Загрузка train")
    train = load_train(TRAIN_FILE)
    print(">> Закончили загрузку данных")

    print(">> Разбиение train")
    train = add_qid_in_train(train)
    train = split_train(train, benchmark_ids=set(items["item_id"]))

    print(">> Классификатор обучается")
    fit_data = train[train["split"] == "fit"]
    val_data = train[(train["split"] == "val") & (train["eligible"])]
    service = MicrocatGeneratorService(popularity_service=PopularityGeneratorService(items)).fit(fit_data, items)
    service.save(MICROCAT_MODEL_FILE)

    print(">> Проверка на val")
    predicted = service.predict_microcats(val_data["normed_search_query"].tolist())
    true_microcats = val_data["item_microcat_id"].to_numpy()

    hits_top5 = [true_microcats[i] in predicted[i] for i in range(len(true_microcats))]
    hits_top1 = [true_microcats[i] == predicted[i][0] for i in range(len(true_microcats))]

    print(f"Строк для проверки использовалось: {len(true_microcats)}")
    print(f"Accuracy@5: {np.mean(hits_top5):.4f}")
    print(f"Accuracy@1: {np.mean(hits_top1):.4f}")
