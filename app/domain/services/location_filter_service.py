import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from app.infrastucture.config import BENCHMARK_I_FILE, BENCHMARK_Q_FILE, TRAIN_FILE
from app.infrastucture.loading import load_bench_items, load_bench_queries, load_train


class LocationFilterService:
    def __init__(self, items: pd.DataFrame, train: pd.DataFrame = None, radius_km: float = 50):
        """Строит индекс локаций по координатам объявлений бенчмарка"""
        self.radius_km = radius_km
        self.loc_centers = (
            items.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .dropna()
        )
        # Ищем координаты локаций запросов из train, тк не у всех в бенч они есть
        if train is not None:
            train_centers = (
                train.groupby("item_location_id")[["item_latitude", "item_longitude"]]
                .median()
                .dropna()
            )
            missings = train_centers.index.difference(self.loc_centers.index)
            self.loc_centers = pd.concat([self.loc_centers, train_centers.loc[missings]])

        coords_radians = np.radians(self.loc_centers.to_numpy())
        self.tree = BallTree(coords_radians, metric="haversine")
        self.item_ids_by_loc = items.groupby("item_location_id")["item_id"].apply(list).to_dict()
        self.all_item_ids = items["item_id"].to_list()

    def _get_neighbor_locs(self, location_id: str, r: float) -> list:
        """Возвращает список локаций в пределах радиуса"""
        if location_id not in self.loc_centers.index:
            return []

        center = self.loc_centers.loc[location_id].to_numpy()
        center_rad = np.radians(center).reshape(1, -1)

        # возвращает индексы точек в пределах радиуса
        radius_rad = r / 6371
        indices = self.tree.query_radius(center_rad, r=radius_rad)[0]

        return self.loc_centers.index[indices].tolist()

    def get_pool(self, location_id: str, is_delivery: bool, min_size: int = 300) -> list:
        if is_delivery:
            return self.all_item_ids
        for r in [self.radius_km, 150, 500, 2000]:
            neighbor_locs = self._get_neighbor_locs(location_id, r)
            pool = []
            for loc in neighbor_locs:
                pool.extend(self.item_ids_by_loc.get(loc, []))
            if len(pool) >= min_size:
                return pool
        return self.all_item_ids

if __name__ == "__main__":
    print(">> Загрузка items")
    items = load_bench_items(BENCHMARK_I_FILE)
    print(">> Загрузка закончена")
    print(">> Загрузка queries")
    queries = load_bench_queries(BENCHMARK_Q_FILE)
    print(">> Загрузка закончена")
    print(">> Загрузка train")
    train = load_train(TRAIN_FILE)
    print(">> Загрузка закончена")

    service = LocationFilterService(items, train=train)
    test_loc = queries["search_location_id"].iloc[0]
    print(f"тест локация: {test_loc}")

    neighbors = service._get_neighbor_locs(test_loc, 50)
    print(f"число соседних локаций (включая саму себя): {len(neighbors)}")
    print(f"первые несколько: {neighbors[:5]}")

    pool = service.get_pool(test_loc, is_delivery=False)
    print(f"размер пула (без доставки): {len(pool)}")

    pool_delivery = service.get_pool(test_loc, is_delivery=True)
    print(f"размер пула (доставка): {len(pool_delivery)}")
    assert len(pool_delivery) == len(items), "при доставке пул должен быть равен всему корпусу"

    print("\nразмеры пулов для первых 10 запросов бенчмарка:")
    for loc in queries["search_location_id"].head(10):
        p = service.get_pool(loc, is_delivery=False)
        print(f"локация {loc}: {len(p)} объявлений")

    for loc in ["649820", "107620", "107621"]:
        in_index = loc in service.loc_centers.index
        print(f"локация {loc}: есть свой центр (есть объявления в корпусе)? {in_index}")

    # узнаем, у какого количество запросов нельзя вычислить локацию
    unique_locs = queries["search_location_id"].unique()
    print(f"Всего уник локаций: {len(unique_locs)} ")

    no_coords_locs = [loc for loc in unique_locs if loc not in service.loc_centers.index]
    print(f"Локаций без известных координат: {len(no_coords_locs)}, ({len(no_coords_locs) / len(unique_locs):.1%}")
    no_coords_mask = queries["search_location_id"].isin(no_coords_locs)
    print(f"Количество запросов бенчмарка без определенных координат: {no_coords_mask.sum()} из {len(queries)} ({no_coords_mask.mean():.1%})")

    # сколько из них с доставкой?
    delivery_share = queries.loc[no_coords_mask, "search_is_delivery_search"].mean()
    print(f"Доля с доставкой: {delivery_share:.1%}")
    print(f"Проблемные без доставки: {(no_coords_mask & (queries["search_is_delivery_search"] == 0)).sum()}")