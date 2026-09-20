import numpy as np
import pandas as pd


class FeatureExtractor:
    """Превращает результаты источников в матрицу признаков для CatBoost.

    Единственное место, где считаются признаки. Вызывается и при сборке
    обучающей выборки, и на инференсе — этим гарантируется, что модель
    видит одинаково устроенные данные в обоих режимах.
    """

    FEATURE_NAMES = [
        # позиции и скоры источников
        "bm25_rank", "bm25_score", "bm25_found",
        "e5_rank", "e5_cos", "e5_found",
        "frida_rank", "frida_cos", "frida_found",
        "microcat_rank", "microcat_found",
        "rrf_rank", "rrf_score",
        # тематика
        "microcat_proba", "is_top1_microcat", "microcat_pos_in_pred",
        # свойства объявления
        "reviews_count", "rating", "popularity_rank",
        # география
        "distance_km", "same_location",
        # контекст запроса
        "pool_size", "n_candidates", "query_len_tokens",
    ]

    def __init__(self, all_items: pd.DataFrame, location_service, popularity_service):
        """all_items — объединение корпуса и объявлений train (515 895 строк)."""

        # Индексируем по item_id, чтобы доставать свойства кандидатов
        # векторно через .reindex(). Поштучный поиск через .loc[] по
        # 700 кандидатам на запрос × 80k запросов — это часы впустую.
        self.meta = all_items.set_index("item_id")[
            ["item_rating_reviews_count", "item_rating",
             "item_microcat_id", "item_latitude", "item_longitude", "item_location_id"]
        ]

        self.location_service = location_service
        self.popularity_service = popularity_service
        # Центры локаций — в словарь {loc_id: (lat, lon)}.
        # В сервисе они лежат DataFrame'ом (loc_centers), но .loc[] по нему
        # внутри цикла на 60k запросов заметно дороже обращения по ключу.
        centers = location_service.loc_centers
        self.loc_centers = {
            loc: (lat, lon)
            for loc, lat, lon in zip(
                centers.index,
                centers["item_latitude"].to_numpy(dtype=np.float64),
                centers["item_longitude"].to_numpy(dtype=np.float64),
            )
        }

    def extract(
        self,
        candidates: list,
        source_rankings: dict,
        source_scores: dict,
        query_location_id,
        predicted_microcats,
        microcat_probas: dict,
        pool_size: int,
        query_text: str,
    ) -> pd.DataFrame:
        """Считает признаки для всех кандидатов одного запроса."""

        n = len(candidates)
        # .reindex() возвращает строки в порядке candidates; для id,
        # которых нет в таблице, ставит NaN — это само по себе защита
        # от рассинхрона данных.
        meta = self.meta.reindex(candidates)

        features = {}

        # --- позиции в источниках ---
        # Позиция — главный сигнал. Кодируем как словарь {item_id: место},
        # чтобы не искать линейно по списку для каждого кандидата.
        for name, key in [("bm25", "bm25"), ("e5", "e5"), ("frida", "frida"),
                          ("microcat", "microcat"), ("rrf", "rrf")]:
            ranked = source_rankings.get(key, [])
            pos_by_id = {item_id: pos for pos, item_id in enumerate(ranked, start=1)}

            ranks = np.array([pos_by_id.get(c, np.nan) for c in candidates], dtype=np.float32)
            features[f"{name}_rank"] = ranks

            # Флаг «источник этого кандидата вообще не вернул».
            # Сам ранг при этом NaN, а НЕ 999 и не -1: CatBoost умеет
            # обрабатывать пропуски нативно (отправляет их в отдельную
            # ветку дерева), а любое число модель проинтерпретирует как
            # осмысленную позицию и сделает неверный вывод.
            if name != "rrf":   # RRF возвращает всех кандидатов, флаг не нужен
                features[f"{name}_found"] = (~np.isnan(ranks)).astype(np.int8)

        # --- сырые скоры ---
        # Ранг теряет масштаб: первое место со скором 12.0 и первое место
        # со скором 0.8 — разные ситуации. Скор это восстанавливает.
        for name, col in [("bm25", "bm25_score"), ("e5", "e5_cos"),
                          ("frida", "frida_cos"), ("rrf", "rrf_score")]:
            scores = source_scores.get(name, {})
            features[col] = np.array(
                [scores.get(c, np.nan) for c in candidates], dtype=np.float32
            )

        # --- тематика ---
        cand_microcats = meta["item_microcat_id"].to_numpy()

        # Вероятность подкатегории этого кандидата по мнению классификатора.
        # Не «совпала/не совпала», а именно число — модель сама решит,
        # с какого порога доверять.
        features["microcat_proba"] = np.array(
            [microcat_probas.get(mc, 0.0) for mc in cand_microcats], dtype=np.float32
        )

        top1 = predicted_microcats[0] if len(predicted_microcats) else None
        features["is_top1_microcat"] = (cand_microcats == top1).astype(np.int8)

        # На каком месте среди топ-5 предсказанных стоит подкатегория кандидата.
        pos_of_microcat = {mc: pos for pos, mc in enumerate(predicted_microcats, start=1)}
        features["microcat_pos_in_pred"] = np.array(
            [pos_of_microcat.get(mc, np.nan) for mc in cand_microcats], dtype=np.float32
        )

        # --- свойства объявления ---
        features["reviews_count"] = meta["item_rating_reviews_count"].to_numpy(dtype=np.float32)
        features["rating"] = meta["item_rating"].to_numpy(dtype=np.float32)
        features["popularity_rank"] = np.array(
            [self.popularity_service.rank_by_id.get(c, np.nan) for c in candidates],
            dtype=np.float32,
        )

        # --- география ---
        center = self.loc_centers.get(query_location_id)
        if center is None:
            # Локация запроса неизвестна — координат нет (17,3% запросов
            # бенчмарка по EDA). NaN честнее любой подстановки.
            features["distance_km"] = np.full(n, np.nan, dtype=np.float32)
        else:
            features["distance_km"] = self._haversine(
                center[0], center[1],
                meta["item_latitude"].to_numpy(dtype=np.float64),
                meta["item_longitude"].to_numpy(dtype=np.float64),
            ).astype(np.float32)
        if center is None:
            features["distance_km"] = np.full(n, np.nan, dtype=np.float32)
        else:
            features["distance_km"] = self._haversine(
                center[0], center[1],
                meta["item_latitude"].to_numpy(dtype=np.float64),
                meta["item_longitude"].to_numpy(dtype=np.float64),
            ).astype(np.float32)

        features["same_location"] = (
            meta["item_location_id"].to_numpy() == query_location_id
        ).astype(np.int8)

        # --- контекст запроса ---
        # Одинаковы для всех кандидатов запроса, но нужны: позволяют модели
        # калибровать остальные признаки. Ранг 50 в пуле из 100 и ранг 50
        # в пуле из 10000 — разного веса. Особенно важно потому, что на
        # обучении пулы строятся из объединения и в среднем крупнее, чем
        # на инференсе.
        features["pool_size"] = np.full(n, pool_size, dtype=np.float32)
        features["n_candidates"] = np.full(n, n, dtype=np.float32)
        features["query_len_tokens"] = np.full(n, len(query_text.split()), dtype=np.float32)

        # Собираем в фиксированном порядке колонок
        return pd.DataFrame(features, columns=self.FEATURE_NAMES, index=candidates)

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Расстояние по поверхности Земли в километрах, векторно."""
        r = 6371.0
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))