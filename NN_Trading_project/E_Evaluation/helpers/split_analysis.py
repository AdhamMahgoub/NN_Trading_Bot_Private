"""
Split-agnostic evaluation utilities for train/val/test analysis.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from .core import buy_metrics, predict_probs_booster

SplitName = Literal["train", "val", "test"]


def _split_bounds(entry: dict, n_rows: int, split: SplitName) -> tuple[int, int]:
    """Return [start, end) bounds for the requested split."""
    if split == "train":
        end = entry.get("train_cut", entry.get("val_start", None))
        if end is None:
            raise KeyError("Index entry missing 'train_cut'/'val_start' for train split.")
        return 0, int(end)

    if split == "val":
        start = entry.get("val_start", entry.get("train_cut", None))
        end = entry.get("val_end", entry.get("test_start", entry.get("val_cut", None)))
        if start is None or end is None:
            raise KeyError("Index entry missing val bounds ('val_start'/'val_end').")
        return int(start), int(end)

    if split == "test":
        start = entry.get("test_start", entry.get("val_cut", None))
        if start is None:
            raise KeyError("Index entry missing 'test_start'/'val_cut' for test split.")
        return int(start), int(n_rows)

    raise ValueError(f"Unsupported split '{split}'. Use one of: train, val, test.")


def _load_model_bundle(model_path: str | Path) -> tuple[xgb.Booster, int]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "booster" not in bundle or "best_ntree" not in bundle:
        raise ValueError(
            "Model bundle must be a dict with keys {'booster', 'best_ntree'}."
        )
    return bundle["booster"], int(bundle["best_ntree"])


def collect_split_predictions(
    split: SplitName,
    index_path: str | Path,
    scaler_path: str | Path,
    booster: xgb.Booster,
    ntree: int,
) -> pd.DataFrame:
    """Collect row-level predictions for one split across all tickers."""
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    rows: list[pd.DataFrame] = []
    for entry in index:
        ticker = entry["ticker"]
        with np.load(entry["cache_file"]) as data:
            X_full = np.asarray(data["X"])
            y_full = np.asarray(data["y"])
            dates_full = data.get("dates", None)

        start, end = _split_bounds(entry, n_rows=len(y_full), split=split)
        start = max(0, start)
        end = min(len(y_full), end)
        if end <= start:
            continue

        X = X_full[start:end].astype(np.float32, copy=False)
        y = y_full[start:end].astype(np.int8, copy=False)
        X_scaled = scaler.transform(X).astype(np.float32, copy=False)
        probs = predict_probs_booster(booster, X_scaled, ntree)

        if dates_full is None:
            dates = pd.Series([pd.NaT] * len(y), dtype="datetime64[ns]")
        else:
            dates = pd.to_datetime(dates_full[start:end], errors="coerce")

        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "ticker": ticker,
                    "y_true": y,
                    "prob_buy": probs.astype(np.float32),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["Date", "ticker", "y_true", "prob_buy", "pred_buy"])

    preds = pd.concat(rows, ignore_index=True)
    preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce").dt.normalize()
    return preds


def analyze_split_predictions(
    preds: pd.DataFrame,
    threshold: float,
    split: SplitName,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-day BUY-trade analysis and summary stats for split predictions."""
    data = preds.copy()
    if data.empty:
        empty_daily = pd.DataFrame(
            columns=["Date", "num_trades", "pct_success", "pct_fail", "num_success", "num_fail"]
        )
        return empty_daily, {
            "split": split,
            "threshold": float(threshold),
            "num_days": 0,
            "total_rows": 0,
            "total_trades": 0,
            "num_success": 0,
            "num_fail": 0,
            "pct_success": 0.0,
            "acc": 0.0,
            "buy_success": 0.0,
            "logloss": np.nan,
        }

    data["pred_buy"] = (data["prob_buy"] >= threshold).astype(np.int8)

    trades = data[data["pred_buy"] == 1].copy()
    trades["is_success"] = (trades["y_true"] == 1).astype(np.int8)
    trades["is_fail"] = (trades["y_true"] == 0).astype(np.int8)

    if trades.empty:
        daily = pd.DataFrame(
            columns=["Date", "num_trades", "pct_success", "pct_fail", "num_success", "num_fail"]
        )
    else:
        daily = (
            trades.groupby("Date", dropna=True)
            .agg(
                num_trades=("pred_buy", "size"),
                num_success=("is_success", "sum"),
                num_fail=("is_fail", "sum"),
            )
            .reset_index()
            .sort_values("Date")
        )
        daily["pct_success"] = 100.0 * daily["num_success"] / daily["num_trades"].clip(lower=1)
        daily["pct_fail"] = 100.0 * daily["num_fail"] / daily["num_trades"].clip(lower=1)

    metrics = buy_metrics(
        y_true=data["y_true"].to_numpy(dtype=np.int64),
        probs=data["prob_buy"].to_numpy(dtype=np.float64),
        threshold=threshold,
    )

    y_vals = data["y_true"].to_numpy(dtype=np.int64)
    p_vals = data["prob_buy"].to_numpy(dtype=np.float64)
    try:
        ll = float(log_loss(y_vals, p_vals, labels=[0, 1]))
    except ValueError:
        ll = np.nan

    total_trades = int(daily["num_trades"].sum()) if not daily.empty else 0
    total_success = int(daily["num_success"].sum()) if not daily.empty else 0
    total_fail = int(daily["num_fail"].sum()) if not daily.empty else 0

    summary = {
        "split": split,
        "threshold": float(threshold),
        "num_days": int(data["Date"].nunique(dropna=True)),
        "total_rows": int(len(data)),
        "total_trades": total_trades,
        "num_success": total_success,
        "num_fail": total_fail,
        "pct_success": 100.0 * total_success / max(1, total_trades),
        "acc": float(metrics["acc"]),
        "buy_success": float(metrics["buy_success"]),
        "logloss": ll,
    }
    return daily, summary


def evaluate_split_from_artifacts(
    split: SplitName,
    threshold: float,
    index_path: str | Path,
    scaler_path: str | Path,
    model_path: str | Path = "best_model_xgb.pkl",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    End-to-end split evaluation from saved artifacts.

    Returns:
        preds: row-level predictions with Date/ticker/y_true/prob_buy/pred_buy
        daily: per-day BUY trade table
        summary: aggregate metrics
    """
    booster, ntree = _load_model_bundle(model_path=model_path)
    preds = collect_split_predictions(
        split=split,
        index_path=index_path,
        scaler_path=scaler_path,
        booster=booster,
        ntree=ntree,
    )
    daily, summary = analyze_split_predictions(preds=preds, threshold=threshold, split=split)
    preds["pred_buy"] = (preds["prob_buy"] >= threshold).astype(np.int8)

    if verbose:
        print(f"Threshold : {threshold:.3f}")
        print(
            f"{split.title()} days: {summary['num_days']}  |  "
            f"Total BUY trades: {summary['total_trades']}"
        )
        print(f"P(success | BUY): {summary['pct_success']:.2f}%")

    return preds, daily, summary
