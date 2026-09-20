import pandas as pd

def normalize_text(s: pd.Series) -> pd.Series:
    """Нормализуем текст запросов и объявлений"""
    return (s.fillna('').astype(str).str.lower()
            .str.replace('ё', 'е', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())

def build_item_text(df: pd.DataFrame, desc_max_chars: int = 500) -> pd.Series:
    """Собирает единый текст объявлений из заголовка, параметров и описания"""
    title = normalize_text(df["item_title_raw"])
    params = normalize_text(df["item_infm_params_text"])

    if "item_description_raw" in df.columns:
        desc = normalize_text(df["item_description_raw"]).str.slice(0, desc_max_chars)
    else:
        desc = pd.Series("", index=df.index)

    return title + " | " + params + " | " + desc