"""
Test-split evaluation helpers that were previously notebook-cell logic.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .core import evaluate_and_save_test


def _to_naive_datetime_index(values: np.ndarray) -> pd.DatetimeIndex:
    dt = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


def _load_intraday_pct_change(csv_path: Path, test_dates: pd.DatetimeIndex, n_test: int) -> list[float]:
    try:
        df_csv = pd.read_csv(csv_path)
    except Exception:
        return [np.nan] * n_test

    if "Date" not in df_csv.columns:
        return [np.nan] * n_test

    df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors="coerce")
    df_csv = df_csv.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df_csv = df_csv[~df_csv.index.duplicated(keep="last")]

    sub = df_csv.reindex(test_dates)
    if "Open" not in sub.columns or "Close" not in sub.columns:
        return [np.nan] * n_test

    sub["Open"] = pd.to_numeric(sub["Open"], errors="coerce")
    sub["Close"] = pd.to_numeric(sub["Close"], errors="coerce")
    pct = ((sub["Close"] - sub["Open"]) / sub["Open"] * 100.0).to_numpy(dtype=np.float64)

    if len(pct) < n_test:
        pct = np.concatenate([pct, np.full(n_test - len(pct), np.nan)])
    else:
        pct = pct[:n_test]
    return pct.tolist()


def build_test_metadata(index_path: str | Path, stocks_dir: str | Path) -> tuple[list[str], list[float]]:
    """Build per-row ticker + same-day percent change for the full test split."""
    index_path = Path(index_path)
    stocks_dir = Path(stocks_dir)

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    test_tickers: list[str] = []
    pct_changes: list[float] = []

    for entry in index:
        ticker = entry["ticker"]
        test_start = entry.get("test_start", entry.get("val_cut", None))
        if test_start is None:
            raise KeyError(f"Index entry for {ticker} missing 'test_start' / 'val_cut'.")
        test_start = int(test_start)

        with np.load(entry["cache_file"]) as data:
            y_full = data["y"]
            dates_full = data.get("dates", None)

        n_test = int(len(y_full)) - test_start
        if n_test <= 0:
            continue

        test_tickers.extend([ticker] * n_test)

        if dates_full is None:
            pct_changes.extend([np.nan] * n_test)
            continue

        test_dates = _to_naive_datetime_index(dates_full[test_start:])
        csv_path = stocks_dir / f"{ticker}.csv"
        pct_changes.extend(_load_intraday_pct_change(csv_path, test_dates, n_test))

    return test_tickers, pct_changes


def evaluate_on_test_data(
    booster: xgb.Booster,
    ntree: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    index_path: str | Path,
    stocks_dir: str | Path,
    save_csv: str | Path = "test_predictions_full.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the notebook's full test-data evaluation flow in one call."""
    test_tickers, pct_changes = build_test_metadata(index_path=index_path, stocks_dir=stocks_dir)

    if len(test_tickers) != len(X_test):
        raise ValueError(f"ticker mismatch: {len(test_tickers)} vs {len(X_test)}")
    if len(pct_changes) != len(X_test):
        raise ValueError(f"pct_change mismatch: {len(pct_changes)} vs {len(X_test)}")

    return evaluate_and_save_test(
        booster=booster,
        ntree=ntree,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        save_csv=save_csv,
        test_tickers=test_tickers,
        pct_changes=pct_changes,
        verbose=verbose,
    )
