"""Small real-model tests for the daily data, backtest, artifacts, and UI."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
import src.train as training
from src.database import load_sales, save_sales
from src.predict import forecast, load_artifact
from src.preprocessing import make_demo_data, validate_and_clean


@pytest.fixture
def paths(tmp_path, monkeypatch):
    values = {
        "RAW_DATA_PATH": tmp_path / "raw" / "retail_sales.csv",
        "DATABASE_PATH": tmp_path / "database" / "sales.db",
        "MODEL_PATH": tmp_path / "models" / "forecaster.joblib",
        "METRICS_PATH": tmp_path / "reports" / "metrics.json",
        "PROCESSED_DATA_PATH": tmp_path / "processed" / "features.csv",
    }
    for name, value in values.items():
        monkeypatch.setattr(training, name, value)
        monkeypatch.setattr(config, name, value)
    return values


def test_daily_contract():
    raw = make_demo_data(days=65, stores=1, products=1)
    assert len(validate_and_clean(pd.concat([raw, raw.iloc[[0]]]))) == len(raw)
    conflict = raw.iloc[[0]].copy()
    conflict.loc[:, "sales"] += 1
    with pytest.raises(ValueError, match="Conflicting rows"):
        validate_and_clean(pd.concat([raw, conflict]))
    with pytest.raises(ValueError, match="Missing calendar date"):
        validate_and_clean(raw.drop(index=10))
    raw.loc[0, "sales"] = 0
    assert validate_and_clean(raw).iloc[0].sales == 0


def test_short_series_is_reported_without_dropping_eligible_series():
    raw = make_demo_data(days=65, stores=1, products=2)
    raw = validate_and_clean(raw[(raw.product_id == 1) | (raw.date >= raw.date.min() + pd.Timedelta(days=25))])
    eligible, excluded = training._eligible(raw, raw.date.max() - pd.Timedelta(days=30))
    assert eligible.product_id.unique().tolist() == [1]
    assert excluded[0]["product_id"] == 2


def test_holdout_sales_change_scores_but_not_predictions():
    raw = validate_and_clean(make_demo_data(days=65, stores=1, products=1))
    first, _, predicted_one = training.backtest(raw)
    changed = raw.copy()
    changed.loc[changed.date > pd.Timestamp(first["cutoff"]), "sales"] += 100
    second, _, predicted_two = training.backtest(changed)
    assert first["models"]["XGBoost"] != second["models"]["XGBoost"]
    for model in ("Prophet", "XGBoost", "Random Forest", "Last observed"):
        pd.testing.assert_frame_equal(predicted_one[model], predicted_two[model])
    assert first["horizon_days"] == first["scored_rows"] == 30


def test_training_models_database_and_horizon_boundaries(paths):
    raw = make_demo_data(days=65, stores=1, products=1)
    report = training.train(frame=raw)
    history = load_sales(paths["DATABASE_PATH"])
    assert len(history) == 65
    assert set(("Prophet", "XGBoost", "Random Forest", "Last observed")).issubset(report["models"])
    artifact = load_artifact(paths["MODEL_PATH"])
    for model in ("Prophet", "XGBoost", "Random Forest"):
        for horizon in (7, 90):
            result = forecast(history, horizon, model, artifact=artifact)
            assert len(result) == horizon
            assert result.date.min() == history.date.max() + pd.Timedelta(days=1)
            assert result.date.max() == history.date.max() + pd.Timedelta(days=horizon)
            assert result[["store_id", "product_id"]].drop_duplicates().values.tolist() == [[1, 1]]
            assert np.isfinite(result.sales).all()
    again = training.train(frame=raw)
    assert again["models"] == report["models"]
    assert len(load_sales(paths["DATABASE_PATH"])) == 65


def test_sqlite_creates_parent_and_replaces_rows(tmp_path):
    path = tmp_path / "new" / "sales.db"
    raw = validate_and_clean(make_demo_data(days=2, stores=1, products=1))
    save_sales(raw, path)
    save_sales(raw.iloc[:1], path)
    assert len(load_sales(path)) == 1


def test_streamlit_first_run_and_model_selection(paths):
    raw = make_demo_data(days=65, stores=1, products=1)
    paths["RAW_DATA_PATH"].parent.mkdir(parents=True)
    raw.to_csv(paths["RAW_DATA_PATH"], index=False)
    app = AppTest.from_file(str(config.BASE_DIR / "dashboard" / "app.py")).run(timeout=120)
    assert not app.exception
    assert not app.error
    assert paths["MODEL_PATH"].exists() and paths["DATABASE_PATH"].exists()
    app.selectbox[0].set_value("Prophet").run(timeout=120)
    assert not app.exception
    assert any("Prophet" in caption.value for caption in app.get("caption"))
    paths["DATABASE_PATH"].unlink()
    recovered = AppTest.from_file(str(config.BASE_DIR / "dashboard" / "app.py")).run(timeout=120)
    assert not recovered.exception
    assert paths["DATABASE_PATH"].exists()


def test_bad_input_keeps_working_artifacts(paths, monkeypatch):
    raw = make_demo_data(days=65, stores=1, products=1)
    training.train(frame=raw)
    before = paths["METRICS_PATH"].read_bytes()
    with pytest.raises(ValueError, match="Missing calendar date"):
        training.train(frame=raw.drop(index=5))
    assert paths["METRICS_PATH"].read_bytes() == before
    assert len(load_sales(paths["DATABASE_PATH"])) == 65
    report = training.backtest(validate_and_clean(raw))[0]
    artifact = load_artifact(paths["MODEL_PATH"])
    monkeypatch.setattr(training, "backtest", lambda frame: (report, frame, {}))
    monkeypatch.setattr(training, "_fit", lambda frame, names: artifact["models"])
    def fail_database(frame, path):
        raise OSError("simulated database staging failure")
    monkeypatch.setattr(training, "save_sales", fail_database)
    with pytest.raises(OSError, match="simulated database staging failure"):
        training.train(frame=raw)
    assert paths["METRICS_PATH"].read_bytes() == before
    assert len(load_sales(paths["DATABASE_PATH"])) == 65


def test_streamlit_bad_source_shows_recovery_message(paths):
    paths["RAW_DATA_PATH"].parent.mkdir(parents=True)
    paths["RAW_DATA_PATH"].write_text("date,store_id,product_id,sales\nnot-a-date,1,1,10\n")
    app = AppTest.from_file(str(config.BASE_DIR / "dashboard" / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.error
    assert "Run python -m src.train" in app.error[0].value
