
"""Сборка обучающей выборки для реранкера.

Прогоняет запросы из train через те же источники, что на инференсе,
размечает кандидатов (выбрали / не выбрали) и считает признаки.
Результат — parquet с колонками FEATURE_NAMES + group_id + label.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd


class TrainsetBuilder:
    def __init__(self, pipeline, n_negatives: int = 30, hard_ratio: float = 0.5,
                 seed: int = 67):
        """pipeline — уже прошедший _setup(), со всеми сервисами.
        n_negatives — сколько негативов оставлять на запрос.
        hard_ratio — доля «сложных» негативов (высоко в RRF, но не выбраны)."""
        self.pipeline = pipeline
        self.n_negatives = n_negatives
        self.hard_ratio = hard_ratio
        self.rng = np.random.default_rng(seed)

    def build(self, queries: pd.DataFrame, gold: dict, out_path: Path,
              top_k: int = 500, batch_size: int = 5000) -> Path:
        """queries — DataFrame с qid и полями запроса.
        gold — {qid: [item_id, ...]} что реально выбирали.
        Пишет частями: 2 млн строк в память не влезут."""

        texts = queries["normed_search_query"].tolist()

        print(">> Подкатегории и кодирование запросов")
        microcats, probas = self.pipeline.microcat_service.predict_microcats_with_proba(texts)
        qvs_e5 = self.pipeline.e5_service.encode_queries(texts)
        qvs_frida = self.pipeline.frida_service.encode_queries(texts)

        print(f">> Сборка выборки: {len(queries)} запросов")
        t0 = time.time()
        buffer = []
        written = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None

        for i, row in enumerate(queries.itertuples()):
            positives = set(gold.get(row.qid, []))
            if not positives:
                continue    # нечему учиться

            out = self.pipeline._run_sources(
                row.normed_search_query, qvs_e5[i], qvs_frida[i], microcats[i],
                row.search_location_id,
                is_delivery=(row.search_is_delivery_search == 1),
                restrict_to=None,       # ОБУЧЕНИЕ: пул из всего объединения,
                top_k=top_k,            # иначе позитивов в кандидатах не будет
            )

            candidates = out["candidates"]
            labels = np.array([1 if c in positives else 0 for c in candidates], dtype=np.int8)

            # Если ни один позитив не попал в кандидаты — запрос бесполезен.
            if labels.sum() == 0:
                continue

            keep = self._sample(candidates, labels, out["rankings"]["rrf"])

            X = self.pipeline.feature_extractor.extract(
                candidates=[candidates[j] for j in keep],
                source_rankings=out["rankings"],
                source_scores=out["scores"],
                query_location_id=row.search_location_id,
                predicted_microcats=microcats[i],
                microcat_probas=probas[i],
                pool_size=len(out["pool"]),
                query_text=row.normed_search_query,
            )
            X["label"] = labels[keep]
            X["group_id"] = row.qid
            buffer.append(X.reset_index(drop=True))

            if len(buffer) >= batch_size:
                written += self._flush(buffer, out_path, first=(written == 0))
                buffer = []
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (len(queries) - i - 1) / speed / 60
                print(f"  {i+1}/{len(queries)} запросов, {written} строк, "
                      f"{speed:.1f} зпр/с, осталось ~{eta:.0f} мин")

        if buffer:
            written += self._flush(buffer, out_path, first=(written == 0))

        print(f">> Готово: {written} строк за {(time.time() - t0)/60:.0f} мин")
        return out_path

    def _sample(self, candidates: list, labels: np.ndarray, rrf_ranked: list) -> np.ndarray:
        """Негативный сэмплинг: все позитивы + n_negatives негативов.

        Смесь случайных и «сложных». Сложные — те, что RRF поставил высоко,
        но пользователь не выбрал: именно их модель путает с позитивами,
        и именно на них она должна учиться. Только случайные дали бы
        слишком лёгкую задачу — модель научилась бы отличать релевантное
        от совсем постороннего, что и без неё умеют источники."""

        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)

        if len(neg_idx) <= self.n_negatives:
            return np.concatenate([pos_idx, neg_idx])

        n_hard = int(self.n_negatives * self.hard_ratio)
        n_random = self.n_negatives - n_hard

        # Сложные: верх RRF-списка, исключая позитивы
        pos_in_rrf = {candidates[j] for j in pos_idx}
        hard_ids = [c for c in rrf_ranked[:100] if c not in pos_in_rrf][:n_hard]
        idx_by_id = {c: j for j, c in enumerate(candidates)}
        hard_idx = np.array([idx_by_id[c] for c in hard_ids if c in idx_by_id], dtype=np.int64)

        # Случайные: из оставшихся
        rest = np.setdiff1d(neg_idx, hard_idx, assume_unique=False)
        n_random = min(n_random + (n_hard - len(hard_idx)), len(rest))
        random_idx = self.rng.choice(rest, size=n_random, replace=False)

        return np.concatenate([pos_idx, hard_idx, random_idx])

    @staticmethod
    def _flush(buffer: list, path: Path, first: bool) -> int:
        """Дописывает батч в parquet."""
        df = pd.concat(buffer, ignore_index=True)
        df.to_parquet(path, index=False,
                      engine="fastparquet", append=not first)
        return len(df)