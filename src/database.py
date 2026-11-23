from __future__ import annotations

import sqlite3
from contextlib import closing
import pandas as pd


def save_sales(frame: pd.DataFrame, database_path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as conn:
        with conn:
            frame.to_sql("sales", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")


def load_sales(database_path) -> pd.DataFrame:
    with closing(sqlite3.connect(database_path)) as conn:
        return pd.read_sql_query("SELECT * FROM sales", conn, parse_dates=["date"])
