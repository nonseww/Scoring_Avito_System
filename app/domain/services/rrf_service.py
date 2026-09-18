from collections import defaultdict


class RRFService:
    """Алгоритм переранжирования списков"""
    def __init__(self, k: int = 60):
        self.k = k

    def run(self, ranked_lists: dict, weights: dict = None, top_k: int = 50) -> list:
        """ranked_lists: {имя_источника: [item_id, ...]} — списки упорядочены по убыванию.
        weights: {имя_источника: вес}, по умолчанию все по 1.0.
        Возвращает top_k item_id по убыванию слитого скора."""
        if weights is None:
            weights = {name: 1.0 for name in ranked_lists}
        scores = defaultdict(float)
        for source, ranked in ranked_lists.items():
            weight = weights.get(source, 1.0)
            if weight == 0:
                continue
            for position, item_id in enumerate(ranked, start=1):
                scores[item_id] += weight / (self.k + position)
        ordered = sorted(scores.items(), key=lambda pair: -pair[1])
        return [item_id for item_id, _ in ordered[:top_k]]