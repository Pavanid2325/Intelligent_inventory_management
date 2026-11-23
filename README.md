# Retail Demand Forecasting and Inventory Analytics

Daily store-product sales forecasting with Prophet, XGBoost, Random Forest, a last-observed baseline, SQLite, Streamlit, pytest, Docker, and GitHub Actions. The bundled demonstration is **synthetic retail data**, generated with a fixed seed. It is suitable for verifying the workflow, not for claiming performance on real retail demand.

## Fresh checkout: Windows PowerShell

Use Python 3.11 (the Docker and CI version). Python 3.12 is also exercised locally, but CI currently validates 3.11 only.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open **http://localhost:8501**. The first app launch also trains the deterministic demo if the artifacts are missing. Training fits all available models twice: once using history up to the backtest cutoff, and again on eligible full history for future use. It can take a few minutes on a first run because Prophet fits a separate model for each store-product pair. Choose **Prophet**, **XGBoost**, or **Random Forest** in the sidebar. The automatic default has the lowest RMSE on the common backtest, which may be the last-observed baseline.

On macOS/Linux, create the environment with `python3.11 -m venv .venv` and run `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

## CSV contract

Put `retail_sales.csv` in `data/raw/`, then run `python -m src.train`, or upload a CSV in the app. Required fields are `date`, `store_id`, `product_id`, and non-negative numeric `sales`. Direct CSV store and product IDs must be integers; the app uploader maps text grouping fields to stable numeric IDs for that upload. Optional `promotion`, `holiday`, and `customers` are accepted. The uploader can map alternate date and metric columns and up to two grouping columns. `Store`, `Product`, `Promo`, and related common names are normalized. Uploading another dataset replaces the previous working dataset only after validation and model fitting complete.

One row means **one observed calendar day for one store-product pair**. Exact duplicate rows collapse. Conflicting same-day rows are rejected: aggregate transactions to daily sales before importing. Missing dates are rejected with the affected date and series; the pipeline never invents zero demand. Observed `sales=0` remains a valid observation. Dates with a time component, invalid/missing required values, and negative sales are rejected. A series needs at least **31 training days plus 30 holdout days**, ending at the common latest date. Short or unaligned series are excluded and listed in the model report; if none qualify, training explains the minimum history needed.

The default demo contains 180 days for two stores and three products. `sample_data/website_traffic_demo.csv` is a separate **non-retail** upload example: select `date` and `website_visitors`, with no grouping. Its visitor counts must not be described as product demand. Rossmann is store-level, so a placeholder product ID can be used only if the dashboard is described as store-level forecasting for that dataset.

## Evaluation and forecast behavior

The default backtest holds out the **latest 30 calendar days** for each eligible series. All candidates and the last-observed baseline use the same cutoff, series, dates, and scored rows. The fitted models see only data through the cutoff; the tree models recursively use their own earlier predictions to build later lag features. Prophet uses per-series `ds` dates and `y` sales. The holdout is used only for scoring. Results report MAE, RMSE, cutoff, horizon, series count, row count, failed models, and excluded series in `reports/model_metrics.json`. The automatic default minimizes RMSE, even if the baseline wins. After backtesting, deployable models are refit on eligible full history; the report retains the held-out scores.

The app lets you request **7 to 90 future days**. Tree-model lag and rolling features use calendar days. Unknown future `promotion` and `holiday` values are assumed **0**; `customers` is not a model feature. Prophet uses only dates and sales. The app shows these assumptions, the chosen model, series count, historical cutoff, and future date range. The chart sums selected store-product series; the downloadable forecast retains each store and product ID. The reported backtest horizon is distinct from the chosen future horizon.

SQLite (`database/sales.db`) holds **cleaned historical sales**, not forecasts. Training replaces its `sales` table on a successful run; rerunning with the same input produces the same demo and report. `models/demand_forecaster.joblib` stores every fitted model, and the app rebuilds missing or legacy artifacts from the saved raw CSV or synthetic demo. Delete generated raw CSV if you want to return to the bundled demo after using your own data.

## Docker

Start Docker Desktop's Linux engine, then run:

```powershell
docker build -t retail-demand-forecast .
docker run --rm -p 8501:8501 retail-demand-forecast
```

Open **http://localhost:8501**. The image excludes the local virtual environment, caches, generated data, SQLite database, and model files. Its first app launch generates and trains the synthetic demo. Data and artifacts created inside this container are **ephemeral** and disappear when it is removed. For persistence, mount the four writable paths (`data/raw`, `data/processed`, `database`, `models`) plus `reports`; create those host directories first, and give the container write permission. A plain `docker run` is the simplest reproducible interview demo.

## Automated testing and provenance

Run `.\.venv\Scripts\python.exe -m pytest -q`. GitHub Actions runs pytest on push and pull request using Python 3.11. The tests include real Prophet, XGBoost, and Random Forest fits, recursive holdout leakage checks, SQLite repeat writes, future horizons, and the Streamlit first-run path.

The repository contains earlier forecasting, SQLite, Streamlit, Docker, and CI code. This implementation adds the daily data contract, Prophet integration, comparable recursive backtest and baseline, persisted model selection, artifact recovery, focused tests, Docker build context exclusions, and this reproduction guide. Available local Git state does not establish authorship of the earlier code; preserve upstream attribution where it can be verified.
