from typing import Optional, List, Literal
import math
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from app.data import load_clean_data, LEVEL_ORDER

app = FastAPI(title="Social media addiction API", version="1.0.0")

data = load_clean_data()
custom_users: List[dict] = []
slope, intercept = np.polyfit(data["total_minutes"], data["addiction_score"], 1)
out_cols = ["country", "age", "age_group", "tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_usage_ratio", "sleep_hours", "heavy_user", "addiction_score", "addiction_level"]


def records(df):
    rows = df[out_cols].astype(object).to_dict(orient="records")
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
            elif k in ("age_group", "addiction_level"):
                r[k] = str(v)
    return rows


def band(score):
    if score < 40:
        return "Low"
    if score < 58:
        return "Medium"
    if score < 70:
        return "High"
    return "Severe"


class UserIn(BaseModel):
    age: int = Field(..., ge=10, le=100, example=21)
    country: str = Field("Unknown", example="Italy")
    tiktok_minutes_daily: float = Field(..., ge=0, example=150)
    instagram_minutes_daily: float = Field(..., ge=0, example=90)
    night_usage_ratio: float = Field(0.5, ge=0, le=1, example=0.6)
    sleep_hours: float = Field(7.0, ge=0, le=24, example=6.5)


class UserOut(UserIn):
    id: int
    total_minutes: float
    daily_hours: float
    predicted_addiction_score: float
    predicted_level: str


@app.get("/")
def root():
    return {"name": "Social media addiction API", "users": int(len(data)), "endpoints": ["GET /users", "GET /stats", "POST /users", "GET /users/custom", "/docs"]}


@app.get("/users")
def get_users(country: Optional[str] = Query(None), level: Optional[str] = Query(None), min_age: Optional[int] = Query(None, ge=0), max_age: Optional[int] = Query(None, ge=0), min_minutes: Optional[float] = Query(None, ge=0), heavy_only: Optional[bool] = Query(None), limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    df = data
    if country is not None:
        df = df[df["country"].str.contains(country, case=False, na=False)]
    if level is not None:
        if level not in LEVEL_ORDER:
            raise HTTPException(400, "level must be one of " + str(LEVEL_ORDER))
        df = df[df["addiction_level"] == level]
    if min_age is not None:
        df = df[df["age"] >= min_age]
    if max_age is not None:
        df = df[df["age"] <= max_age]
    if min_minutes is not None:
        df = df[df["total_minutes"] >= min_minutes]
    if heavy_only is not None:
        df = df[df["heavy_user"] == (1 if heavy_only else 0)]
    page = df.iloc[offset:offset + limit]
    return {"total": int(len(df)), "offset": offset, "limit": limit, "count": int(len(page)), "items": records(page)}


@app.get("/stats")
def get_stats(group_by: Literal["addiction_level", "age_group", "country", "heavy_user"] = Query("addiction_level"), metric: Literal["avg_addiction", "avg_total_minutes", "avg_sleep", "count"] = Query("avg_addiction")):
    g = data.groupby(group_by, observed=True)
    if metric == "avg_addiction":
        res = g["addiction_score"].mean().round(2)
    elif metric == "avg_total_minutes":
        res = g["total_minutes"].mean().round(2)
    elif metric == "avg_sleep":
        res = g["sleep_hours"].mean().round(2)
    else:
        res = g.size()
    res = res.sort_values(ascending=False)
    return {"group_by": group_by, "metric": metric, "result": {str(k): float(v) for k, v in res.items()}}


@app.post("/users", response_model=UserOut, status_code=201)
def create_user(u: UserIn):
    total = u.tiktok_minutes_daily + u.instagram_minutes_daily
    pred = float(np.clip(intercept + slope * total, 0, 100).round(2))
    rec = UserOut(id=len(custom_users) + 1, total_minutes=total, daily_hours=round(total / 60, 2), predicted_addiction_score=pred, predicted_level=band(pred), **u.model_dump())
    custom_users.append(rec.model_dump())
    return rec


@app.get("/users/custom")
def list_custom():
    return {"count": len(custom_users), "items": custom_users}
