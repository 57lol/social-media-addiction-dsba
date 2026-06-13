import os
import functools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "data.csv")
LEVEL_ORDER = ["Low", "Medium", "High", "Severe"]


@functools.lru_cache(maxsize=1)
def load_clean_data(csv_path=CSV_PATH):
    data = pd.read_csv(csv_path)
    data["addiction_level"] = pd.Categorical(data["addiction_level"], categories=LEVEL_ORDER, ordered=True)
    data = data.drop(columns=["user_id"], errors="ignore")
    data["total_minutes"] = data["tiktok_minutes_daily"] + data["instagram_minutes_daily"]
    data["daily_hours"] = data["total_minutes"] / 60
    data["tiktok_share"] = data["tiktok_minutes_daily"] / data["total_minutes"].replace(0, np.nan)
    data["night_minutes"] = data["total_minutes"] * data["night_usage_ratio"]
    data["age_group"] = pd.cut(data["age"], bins=[0, 25, 35, 200], labels=["<25", "25-35", "35+"])
    data["heavy_user"] = (data["total_minutes"] > data["total_minutes"].median()).astype(int)
    data["level_num"] = data["addiction_level"].cat.codes + 1
    for c in ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_minutes", "sleep_hours", "addiction_score"]:
        data[c + "_z"] = (data[c] - data[c].mean()) / data[c].std()
    return data


NUMERIC_FIELDS = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours", "addiction_score", "night_usage_ratio", "attention_span_score"]
