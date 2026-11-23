"""Forecast every eligible daily store-product series from one origin."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH
from src.feature_engineering import FEATURE_COLUMNS

KEYS = ["store_id", "product_id"]
BASELINE = "Last observed"


def load_artifact(path=MODEL_PATH) -> dict:
    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise ValueError("Forecast artifacts are unavailable. Run python -m src.train to rebuild them.") from error
    if artifact.get("version") != 2 or "models" not in artifact:
        raise ValueError("Legacy forecast artifact detected. Run python -m src.train to rebuild it.")
    return artifact


def forecast(history: pd.DataFrame, days: int = 30, model_name: str | None = None,
             artifact_path=MODEL_PATH, artifact: dict | None = None) -> pd.DataFrame:
    """Use historical sales and future promo=holiday=0 for unknown dates."""
    if not 1 <= days <= 90:
        raise ValueError("Forecast horizon must be between 1 and 90 days.")
    artifact = artifact if artifact is not None else load_artifact(artifact_path)
    name = model_name or artifact["selected_model"]
    if name not in artifact["models"] and name != BASELINE:
        raise ValueError(f"Model {name!r} is unavailable. Rebuild artifacts or choose an available model.")
    working = history.copy()
    working["date"] = pd.to_datetime(working["date"])
    if working.empty:
        raise ValueError("No historical sales are available for the selected series.")
    eligible = {tuple(pair) for pair in artifact["eligible_series"]}
    groups = {key: group.sort_values("date") for key, group in working.groupby(KEYS)}
    groups = {key: group for key, group in groups.items() if key in eligible}
    if not groups:
        raise ValueError("Selected series have insufficient training history. Choose an eligible store and product.")
    origin = working.date.max()
    if any(group.date.max() != origin for group in groups.values()):
        raise ValueError("Selected series must end at the same forecast origin. Rebuild with aligned daily data.")
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=days, freq="D")
    rows = []
    if name == "Prophet":
        for key, group in groups.items():
            model = artifact["models"][name][key]
            predicted = model.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy()
            rows.extend((day, *key, max(0.0, float(value))) for day, value in zip(dates, predicted))
    elif name == BASELINE:
        for key, group in groups.items():
            rows.extend((day, *key, float(group.sales.iloc[-1])) for day in dates)
    else:
        model = artifact["models"][name]
        past = {key: group.sales.astype(float).tolist() for key, group in groups.items()}
        for day in dates:
            features = []
            for (store, product), values in past.items():
                if len(values) < 30:
                    raise ValueError(f"Store {store}, product {product} needs 30 prior daily sales for lag features.")
                features.append({
                    "store_id": store, "product_id": product,
                    "year": day.year, "month": day.month, "quarter": day.quarter,
                    "day_of_week": day.day_of_week, "weekend": int(day.day_of_week >= 5),
                    "promotion": 0, "holiday": 0,
                    "lag_1": values[-1], "lag_7": values[-7], "lag_30": values[-30],
                    "rolling_mean_7": float(np.mean(values[-7:])),
                    "rolling_mean_14": float(np.mean(values[-14:])),
                    "rolling_mean_30": float(np.mean(values[-30:])),
                })
            predictions = np.maximum(model.predict(pd.DataFrame(features)[FEATURE_COLUMNS]), 0)
            for key, value in zip(past, predictions):
                past[key].append(float(value))
                rows.append((day, *key, float(value)))
    result = pd.DataFrame(rows, columns=["date", "store_id", "product_id", "sales"])
    if not np.isfinite(result.sales.to_numpy()).all():
        raise ValueError(f"{name} produced a non-finite forecast. Rebuild artifacts or inspect input data.")
    return result.sort_values(["date", *KEYS]).reset_index(drop=True)
