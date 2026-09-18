import pandas as pd


class PopularityGeneratorService:
    def __init__(self, items: pd.DataFrame):
        """Отдает объявления, отсортированные по репутации:
        берем байесовское сглаживание рейтинга через количество отзывов
        и рейтинг"""
        