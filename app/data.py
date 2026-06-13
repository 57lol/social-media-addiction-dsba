# -*- coding: utf-8 -*-
"""
Shared data loading + feature engineering for the Social Media Addiction project.
Mirrors the notebook pipeline; used by both the FastAPI API and the Streamlit app.
"""
import os
import functools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "data.csv")

LEVEL_ORDER = ["Low", "Medium", "High", "Severe"]


@functools.lru_cache(maxsize=1)
def load_clean_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Loads the dataset, fixes types and adds engineered features. Cached."""
    data = pd.read_csv(csv_path)

    # the data is already clean (no NaN / no duplicates); fix the categorical type
    data["addiction_level"] = pd.Categorical(
        data["addiction_level"], categories=LEVEL_ORDER, ordered=True)
    data = data.drop(columns=["user_id"], errors="ignore")

    # engineered features
    data["total_minutes"] = data["tiktok_minutes_daily"] + data["instagram_minutes_daily"]
    data["daily_hours"]   = data["total_minutes"] / 60
    data["tiktok_share"]  = data["tiktok_minutes_daily"] / data["total_minutes"].replace(0, np.nan)
    data["night_minutes"] = data["total_minutes"] * data["night_usage_ratio"]
    data["age_group"]     = pd.cut(data["age"], bins=[0, 25, 35, 200],
                                   labels=["<25", "25-35", "35+"])
    data["heavy_user"]    = (data["total_minutes"] > data["total_minutes"].median()).astype(int)
    data["level_num"]     = data["addiction_level"].cat.codes + 1

    return data


NUMERIC_FIELDS = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours",
                  "addiction_score", "night_usage_ratio", "attention_span_score"]
