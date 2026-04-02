"""
Core XGBoost evaluation helpers shared by notebooks/scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb


def predict_probs_booster(booster: xgb.Booster, X: np.ndarray, ntree: int) -> np.ndarray:
    """Version-safe probability prediction from a native XGBoost booster."""
    dmat = xgb.DMatrix(X)
    try:
        probs = booster.predict(dmat, iteration_range=(0, ntree))
    except TypeError:
        probs = booster.predict(dmat, ntree_limit=ntree)
    return np.asarray(probs, dtype=np.float32).ravel()


def buy_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Confusion matrix + accuracy + P(success | BUY) at a threshold."""
    pred = (np.asarray(probs).ravel() >= threshold).astype(np.int64)
    y = np.asarray(y_true).ravel().astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "acc": 100.0 * (tp + tn) / max(1, len(y)),
        "buy_success": 100.0 * tp / max(1, tp + fp),
    }


def evaluate_and_save_test(
    booster: xgb.Booster,
    ntree: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    save_csv: str | Path = "test_predictions_full.csv",
    test_tickers: Sequence[str] | None = None,
    pct_changes: Sequence[float] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run test evaluation, save prediction CSV, and return predictions + summary."""
    probs = predict_probs_booster(booster, X_test, ntree)
    pred = (probs >= threshold).astype(np.int64)
    ytrue = np.asarray(y_test).ravel().astype(np.int64)

    df = pd.DataFrame(
        {
            "idx": np.arange(len(ytrue), dtype=np.int64),
            "prob_buy": probs.astype(np.float32),
            "pred": pred,
            "actual": ytrue,
            "correct": pred == ytrue,
            "decision": np.where(pred == 1, "BUY", "NO-BUY"),
            "actual_label": np.where(ytrue == 1, "BUY", "NO-BUY"),
        }
    )

    if test_tickers is not None and len(test_tickers) == len(df):
        df["ticker"] = list(test_tickers)
    if pct_changes is not None and len(pct_changes) == len(df):
        df["pct_change"] = np.asarray(pct_changes, dtype=np.float64)

    save_path = Path(save_csv)
    df.to_csv(save_path, index=False)

    metrics = buy_metrics(ytrue, probs, threshold)
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    n = max(1, len(ytrue))
    summary_df = pd.DataFrame(
        {
            "category": [
                "BUY_success_TP",
                "BUY_fail_FP",
                "NO_BUY_success_TN",
                "NO_BUY_fail_FN",
            ],
            "count": [tp, fp, tn, fn],
            "pct_of_all_%": [100 * tp / n, 100 * fp / n, 100 * tn / n, 100 * fn / n],
            "pct_given_decision_%": [
                100 * tp / max(1, tp + fp),
                100 * fp / max(1, tp + fp),
                100 * tn / max(1, tn + fn),
                100 * fn / max(1, tn + fn),
            ],
        }
    )

    if verbose:
        print(f"Saved test predictions -> {save_path}")
        print(
            "P(success | BUY): "
            f"{metrics['buy_success']:.2f}%  |  acc: {metrics['acc']:.2f}%"
        )

    return df, summary_df
