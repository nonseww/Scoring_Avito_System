from collections import defaultdict


class RRFService:
    """Алгоритм переранжирования списков - RRF"""
    def __init__(self, k: int = 60):
        self.k = k

    def run_with_scores(self, ranked_lists: dict, weights: dict = None, top_k: int | None = 50) -> tuple:
        """Возвращает (упорядоченный список, {item_id: слитый скор}).
        top_k=None — вернуть всех кандидатов (нужно для признаков реранкера)"""
        if weights is None:
            weights = {name: 1.0 for name in ranked_lists}

        scores = defaultdict(float)
        for source_name, ranked in ranked_lists.items():
            weight = weights.get(source_name, 1.0)
            if weight == 0:
                continue
            for position, item_id in enumerate(ranked, start=1):
                scores[item_id] += weight / (self.k + position)

        ordered = sorted(scores.items(), key=lambda pair: -pair[1])
        if top_k is not None:
            ordered = ordered[:top_k]
        return [item_id for item_id, _ in ordered], dict(scores)

    def run(self, ranked_lists, weights=None, top_k=50) -> list:
        """Возвращает ответ без score"""
        ids, _ = self.run_with_scores(ranked_lists, weights, top_k)
        return ids