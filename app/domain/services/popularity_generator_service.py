import pandas as pd


class PopularityGeneratorService:
    def __init__(self, items: pd.DataFrame):
        """Отдает объявления, отсортированные по репутации:
        сначала с высоким числом отзывов, потом с высоким рейтингом"""
        self.items_sorted = items.sort_values(
            ["item_rating_reviews_count", "item_rating"],
            ascending=False,
            na_position="last"
        )
        self.rank_by_id = {
            item_id: rank for rank, item_id in enumerate(self.items_sorted["item_id"])
        }

    def generate_popularity_ranking(self, pool: list, top_k: int = 200) -> list:
        pool_sorted = sorted(
            pool, key=lambda item_id: self.rank_by_id.get(item_id, len(self.rank_by_id))
        )
        return pool_sorted[:top_k]

if __name__ == "__main__":
    test_items = pd.DataFrame({
        "item_id": ["aaa", "bbb", "ccc", "ddd", "eee", "fff"],
        "item_rating_reviews_count": [10, 500, 0, 500, None, 50],
        "item_rating": [4.9, 4.2, 5.0, 4.8, 5.0, 4.5],
    })

    service = PopularityGeneratorService(test_items)
    print(service.items_sorted[["item_id", "item_rating_reviews_count", "item_rating"]])
    print(f"rank_by_id: {service.rank_by_id}")

    pool = ["ccc", "aaa", "ddd", "fff"]
    result = service.generate_popularity_ranking(pool, top_k=3)
    print("пул на входе:  ", pool)
    print("топ-3 на выходе:", result)