"""Comparable recursive backtest and fitted deployable models."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from config import DATABASE_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.database import save_sales
from src.evaluation import metrics
from src.feature_engineering import FEATURE_COLUMNS, create_features
from src.predict import BASELINE, forecast
from src.preprocessing import make_demo_data, validate_and_clean

BACKTEST_DAYS = 30
MIN_TRAIN_DAYS = 31


def _models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.05,
                                objective="reg:squarederror", random_state=42, n_jobs=1),
    }


def _fit(history: pd.DataFrame, names: list[str]) -> dict:
    featured = create_features(history).dropna(subset=FEATURE_COLUMNS)
    models = {}
    for name, model in _models().items():
        if name in names:
            model.fit(featured[FEATURE_COLUMNS], featured.sales)
            models[name] = model
    if "Prophet" in names:
        prophets = {}
        for key, group in history.groupby(["store_id", "product_id"]):
            model = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                            daily_seasonality=False, uncertainty_samples=0)
            model.fit(group.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]])
            prophets[key] = model
        models["Prophet"] = prophets
    return models


def _eligible(raw: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    included, excluded = [], []
    origin = raw.date.max()
    for (store, product), group in raw.groupby(["store_id", "product_id"]):
        training = group[group.date <= cutoff]
        if group.date.max() != origin or len(training) < MIN_TRAIN_DAYS or len(group[group.date > cutoff]) != BACKTEST_DAYS:
            excluded.append({"store_id": int(store), "product_id": int(product),
                             "reason": f"Needs {MIN_TRAIN_DAYS} training days and {BACKTEST_DAYS} holdout days ending {origin:%Y-%m-%d}."})
        else:
            included.append(group)
    if not included:
        raise ValueError(f"No eligible series: supply at least {MIN_TRAIN_DAYS + BACKTEST_DAYS} consecutive daily observations per store-product through {origin:%Y-%m-%d}.")
    return pd.concat(included, ignore_index=True), excluded


def backtest(raw: pd.DataFrame, horizon: int = BACKTEST_DAYS) -> tuple[dict, pd.DataFrame, dict]:
    if horizon != BACKTEST_DAYS:
        raise ValueError(f"This pipeline uses a {BACKTEST_DAYS}-day comparable backtest.")
    cutoff = raw.date.max() - pd.Timedelta(days=horizon)
    eligible, excluded = _eligible(raw, cutoff)
    history, holdout = eligible[eligible.date <= cutoff], eligible[eligible.date > cutoff]
    actual = holdout.set_index(["date", "store_id", "product_id"]).sales.sort_index()
    names = [*list(_models()), "Prophet"]
    backtest_models, failures = {}, {}
    for name in names:
        try:
            backtest_models.update(_fit(history, [name]))
        except Exception as error:
            failures[name] = str(error)
    artifact = {"version": 2, "models": backtest_models,
                "eligible_series": list(history.groupby(["store_id", "product_id"]).groups)}
    scores, predictions = {}, {}
    for name in [*backtest_models, BASELINE]:
        try:
            predicted = forecast(history, horizon, name, artifact=artifact)
            values = predicted.set_index(["date", "store_id", "product_id"]).sales.sort_index()
            if not values.index.equals(actual.index):
                raise ValueError("Forecast dates or series differ from the common holdout.")
            scores[name] = metrics(actual.to_numpy(), values.to_numpy())
            predictions[name] = predicted
        except Exception as error:
            failures[name] = str(error)
    if not scores:
        raise ValueError(f"No model completed the common backtest: {failures}")
    best = min(scores, key=lambda name: scores[name]["rmse"])
    report = {
        "selected_model": best, "selection_metric": "rmse",
        "cutoff": cutoff.date().isoformat(), "holdout_start": (cutoff + pd.Timedelta(days=1)).date().isoformat(),
        "horizon_days": horizon, "series_count": len(artifact["eligible_series"]),
        "scored_rows": len(actual), "models": scores, "failures": failures,
        "excluded_series": excluded,
        "future_assumptions": "Promotion and holiday are zero for unknown future dates; customers are unused.",
    }
    return report, eligible, predictions


def _publish(files: list[tuple[Path, object]]) -> None:
    """Stage all outputs first; restore previous files if publication fails."""
    staged, backups = {}, {}
    try:
        for path, writer in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as handle:
                staged[path] = Path(handle.name)
            writer(staged[path])
        for path in staged:
            if path.exists():
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".backup-", delete=False) as handle:
                    backups[path] = Path(handle.name)
                shutil.copy2(path, backups[path])
        replaced = []
        try:
            for path, temporary in staged.items():
                os.replace(temporary, path)
                replaced.append(path)
        except Exception:
            for path in reversed(replaced):
                if path in backups:
                    os.replace(backups[path], path)
                else:
                    path.unlink(missing_ok=True)
            raise
    finally:
        for temporary in [*staged.values(), *backups.values()]:
            temporary.unlink(missing_ok=True)


def train(raw_path=None, frame: pd.DataFrame | None = None) -> dict:
    raw_path = Path(raw_path) if raw_path is not None else RAW_DATA_PATH
    generated = frame is None and not raw_path.exists()
    raw = validate_and_clean(frame if frame is not None else
                             pd.read_csv(raw_path) if raw_path.exists() else make_demo_data(days=180, stores=2, products=3))
    report, eligible, _ = backtest(raw)
    fitted = _fit(eligible, [name for name in report["models"] if name != BASELINE])
    artifact = {"version": 2, "models": fitted,
                "selected_model": report["selected_model"],
                "eligible_series": list(eligible.groupby(["store_id", "product_id"]).groups),
                "horizon_days": BACKTEST_DAYS}
    # Validation and fitting complete before any existing artifact is replaced.
    files = [
        (MODEL_PATH, lambda path: joblib.dump(artifact, path)),
        (METRICS_PATH, lambda path: path.write_text(json.dumps(report, indent=2))),
        (DATABASE_PATH, lambda path: save_sales(raw, path)),
        (PROCESSED_DATA_PATH, lambda path: create_features(eligible).to_csv(path, index=False)),
    ]
    if generated or frame is not None:
        files.append((raw_path, lambda path: raw.to_csv(path, index=False)))
    _publish(files)
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
