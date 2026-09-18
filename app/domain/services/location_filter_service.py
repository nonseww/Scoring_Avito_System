import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


class LocationFilterService:
    def __init__(self, items: pd.DataFrame, radius_km: float = 50):
        """Строит индекс локаций по координатам объявлений бенчмарка"""
        self.radius_km = radius_km
        self.loc_centers = (
            items.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .dropna()
        )
        coords_radians = np.radians(self.loc_centers.to_numpy())
        self.tree = BallTree(coords_radians, metric="haversine")
        self.item_ids_by_loc = items.groupby("item_locationd_id")["item_id"].apply(list).to_dict()
        self.all_item_ids = items["item_id"].to_list()

    def get_neighbor_locs(self, location_id: str) -> list:
        """Возвращает список локаций в пределах радиуса"""
        if location_id not in self.loc_centers.index:
            return []

        center = self.loc_centers.loc[location_id].to_numpy()
        center_rad = np.radians(center).reshape(1, -1)

        # возвращает индексы точек в пределах радиуса
        radius_rad = self.radius_km / 6371
        indices = self.tree.query_radius(center_rad, r=radius_rad)[0]

        return self.loc_centers[indices].tolist()

    def get_pool(self, location_id: str, is_delivery: bool) -> list:
        if is_delivery:
            return self.all_item_ids
        neighbor_locs = self.get_neighbor_locs(location_id)
        pool = []
        for loc in neighbor_locs:
            pool.extend(self.item_ids_by_loc.get(loc, []))
        return pool