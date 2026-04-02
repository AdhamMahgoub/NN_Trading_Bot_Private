"""
Real-life start-of-day ticker scan using the trained XGBoost booster.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from helpers.evaluation.core import predict_probs_booster
from helpers.feature.feature_builder import compute_engineered_features


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts


def build_model_features_from_df(df_upto: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build the same engineered feature set as training (no labels).
    """
    if "Date" not in df_upto.columns:
        raise ValueError("build_model_features_from_df expects a 'Date' column.")

    df = df_upto.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in df_upto for feature building.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open"]).reset_index(drop=True)
    if len(df) < max(window + 2, 2):
        return pd.DataFrame(columns=["Date"])

    # NOTE: shared builder intentionally computes some extra columns that are
    # not currently in FEATURE_COLS; they are retained for experiment parity.
    return compute_engineered_features(df, window=window)


def _extract_one_ticker_df(df_all: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract a single-ticker OHLCV frame from yf.download multi-ticker output."""
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    cols = df_all.columns
    if isinstance(cols, pd.MultiIndex) and ticker in cols.get_level_values(0):
        d = df_all[ticker].copy()
        need = ["Open", "High", "Low", "Close", "Volume"]
        if not set(need).issubset(d.columns):
            return pd.DataFrame()
        return d[need].copy()

    if isinstance(cols, pd.MultiIndex) and ticker in cols.get_level_values(-1):
        need = ["Open", "High", "Low", "Close", "Volume"]
        out = pd.DataFrame(index=df_all.index)
        for fld in need:
            if (fld, ticker) in cols:
                out[fld] = df_all[(fld, ticker)]
            elif (ticker, fld) in cols:
                out[fld] = df_all[(ticker, fld)]
            else:
                return pd.DataFrame()
        return out

    if not isinstance(cols, pd.MultiIndex):
        need = ["Open", "High", "Low", "Close", "Volume"]
        if set(need).issubset(df_all.columns):
            return df_all[need].copy()
    return pd.DataFrame()


def _chunks(values: Sequence[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _maybe_display(df: pd.DataFrame) -> None:
    try:
        from IPython.display import display  # type: ignore

        display(df)
    except Exception:
        print(df)


def run_start_of_day_scan(
    tickers: Iterable[str],
    window: int,
    buy_threshold: float,
    *,
    profit_threshold: float | None = None,
    horizon_bars: int | None = None,
    interval: str = "1d",
    lookback_days: int | None = None,
    chunk_size: int = 40,
    sleep_between_chunks: float = 0.5,
    cache_dir: str | Path | None = None,
    model_path: str | Path = "best_model_xgb.pkl",
    market_timezone: str = "America/New_York",
    now: pd.Timestamp | None = None,
    save_csv: bool | str | Path = True,
    print_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """
    Scan tickers at market open and decide BUY/NO BUY from the trained booster.
    """
    window = int(window)
    prob_threshold = float(buy_threshold)
    lookback_days = int(lookback_days) if lookback_days is not None else max(window * 3, 180)
    model_path = Path(model_path)
    if cache_dir is None:
        cache_dir = Path(f".feature_cache_forward_return_w{window}")
    else:
        cache_dir = Path(cache_dir)
    scaler_path = cache_dir / "scaler.pkl"

    print("Using hyperparameters:")
    print(f"  BUY_THRESHOLD:      {buy_threshold}")
    print(f"  PROFIT_THRESHOLD:   {profit_threshold}")
    print(f"  HORIZON_BARS:       {horizon_bars}")
    print(f"  WINDOW:             {window}")
    print(f"  INTERVAL:           {interval}")
    print(f"  LOOKBACK_DAYS:      {lookback_days}")
    print()

    assert scaler_path.exists(), f"Missing scaler at: {scaler_path}"
    assert model_path.exists(), f"Missing XGBoost model bundle at: {model_path}"

    with open(scaler_path, "rb") as f:
        scaler_loaded = pickle.load(f)

    bundle = joblib.load(model_path)
    assert isinstance(bundle, dict) and "booster" in bundle and "best_ntree" in bundle, (
        "best_model_xgb.pkl must be a dict bundle: {'booster': Booster, 'best_ntree': int}"
    )
    booster = bundle["booster"]
    best_ntree = int(bundle["best_ntree"])

    seen = set()
    tickers_uniq: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            tickers_uniq.append(ticker)
            seen.add(ticker)

    tz_market = ZoneInfo(market_timezone)
    now_market = pd.Timestamp.now(tz_market) if now is None else pd.Timestamp(now)
    if now_market.tzinfo is None:
        now_market = now_market.tz_localize(tz_market)
    else:
        now_market = now_market.tz_convert(tz_market)

    today_market = now_market.normalize()
    start_market = today_market - pd.Timedelta(days=lookback_days)
    end_market = today_market + pd.Timedelta(days=1)
    start_naive_utc = _to_naive_utc(start_market)
    end_naive_utc = _to_naive_utc(end_market)

    def predict_prob_buy_from_history(
        df_upto: pd.DataFrame, today_open_price: float | None = None
    ) -> dict:
        if df_upto is None or df_upto.empty:
            return {"ok": False, "reason": "no history"}

        last_complete_time = df_upto.index[-1]
        last_complete_row = df_upto.iloc[-1]
        tmp = df_upto.reset_index().rename(columns={df_upto.index.name or "index": "Date"})

        if today_open_price is not None:
            today_date = last_complete_time + pd.Timedelta(days=1)
            today_row = pd.DataFrame(
                {
                    "Date": [today_date],
                    "Open": [today_open_price],
                    "High": [np.nan],
                    "Low": [np.nan],
                    "Close": [np.nan],
                    "Volume": [np.nan],
                }
            )
            tmp = pd.concat([tmp, today_row], ignore_index=True)

        feats = build_model_features_from_df(tmp, window)
        if feats.empty:
            return {"ok": False, "reason": "features NaN (insufficient warmup)"}

        last = feats.iloc[-1]
        X_row = last.drop(labels=["Date"]).to_numpy(dtype=np.float32, copy=False).reshape(1, -1)
        Xs = scaler_loaded.transform(X_row).astype(np.float32, copy=False)
        prob_buy = float(predict_probs_booster(booster, Xs, best_ntree)[0])

        return {
            "ok": True,
            "bar_used_market": last_complete_time,
            "prob_buy": prob_buy,
            "cutoff_open": float(last_complete_row["Open"]),
            "cutoff_high": float(last_complete_row["High"]),
            "cutoff_low": float(last_complete_row["Low"]),
            "cutoff_close": float(last_complete_row["Close"]),
            "cutoff_volume": float(last_complete_row["Volume"]),
            "today_open": today_open_price if today_open_price is not None else None,
        }

    results: list[dict] = []
    errors: list[dict] = []

    print(f"Scanning {len(tickers_uniq)} tickers at START OF DAY")
    print(f"Market timezone: {tz_market}, today: {today_market}")
    print(f"Downloading history from {start_market} to {end_market} (market time) ...")
    print("This will get ALL previous complete bars plus TODAY'S opening price\n")

    for chunk in _chunks(tickers_uniq, max(1, int(chunk_size))):
        try:
            df_all = yf.download(
                tickers=chunk,
                start=start_naive_utc,
                end=end_naive_utc,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            for ticker in chunk:
                errors.append({"ticker": ticker, "error": f"download failed: {exc}"})
            time.sleep(max(0.0, float(sleep_between_chunks)))
            continue

        if df_all is None or df_all.empty:
            for ticker in chunk:
                errors.append({"ticker": ticker, "error": "empty download"})
            time.sleep(max(0.0, float(sleep_between_chunks)))
            continue

        idx = pd.to_datetime(df_all.index, errors="coerce")
        df_all = df_all.copy()
        df_all.index = idx
        df_all = df_all[~df_all.index.isna()]

        if df_all.index.tz is None:
            df_all.index = df_all.index.tz_localize(tz_market)
        else:
            df_all.index = df_all.index.tz_convert(tz_market)

        for ticker in chunk:
            d = _extract_one_ticker_df(df_all, ticker)
            if d.empty:
                errors.append({"ticker": ticker, "error": "no usable OHLCV in download"})
                continue

            d = d.sort_index()
            d_complete = d.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            today_bars = d[d.index >= today_market]
            today_open_price = None

            if not today_bars.empty and not pd.isna(today_bars.iloc[0]["Open"]):
                today_open_price = float(today_bars.iloc[0]["Open"])

            if today_open_price is None:
                print(
                    f"WARNING: {ticker} - today's opening price not found. "
                    "Market may not be open yet or data is incomplete. Skipping."
                )
                errors.append({"ticker": ticker, "error": "today's opening price not available"})
                continue

            d_history = d_complete[d_complete.index < today_market]
            if d_history.empty:
                errors.append({"ticker": ticker, "error": "no complete historical bars"})
                continue

            res = predict_prob_buy_from_history(d_history, today_open_price)
            if not res["ok"]:
                errors.append({"ticker": ticker, "error": res["reason"]})
                continue

            p = res["prob_buy"]
            decision = "BUY" if p >= prob_threshold else "NO BUY"

            result_data = {
                "ticker": ticker,
                "bar_used_market": res["bar_used_market"],
                "prob_buy": p,
                "prob_buy_%": 100.0 * p,
                "decision": decision,
                "yesterday_close": res["cutoff_close"],
            }
            result_data["today_open"] = today_open_price
            result_data["gap_%"] = 100.0 * (today_open_price / res["cutoff_close"] - 1.0)
            results.append(result_data)

        time.sleep(max(0.0, float(sleep_between_chunks)))

    if results:
        df_res = (
            pd.DataFrame(results)
            .sort_values(["decision", "prob_buy"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        df_res = pd.DataFrame(
            columns=[
                "ticker",
                "bar_used_market",
                "prob_buy",
                "prob_buy_%",
                "decision",
                "yesterday_close",
                "today_open",
                "gap_%",
            ]
        )

    df_err = pd.DataFrame(errors)
    if not df_err.empty and "ticker" in df_err.columns:
        df_err = df_err.sort_values("ticker").reset_index(drop=True)

    print("\n================ RESULTS ================")
    print(f"OK: {len(df_res)} tickers | Errors: {len(df_err)} tickers")
    print("\nNOTE: This scan is designed for START OF DAY trading.")
    print("      Features use YESTERDAY's complete bar + TODAY's opening price (if available).")
    print("      If 'today_open' column is present, gap_% shows overnight gap.\n")
    if print_tables:
        _maybe_display(df_res)

    if not df_err.empty:
        print("\n================ ERRORS (top 50) ================")
        if print_tables:
            _maybe_display(df_err.head(50))

    out_csv_path: Path | None = None
    if save_csv:
        if isinstance(save_csv, (str, Path)):
            out_csv_path = Path(save_csv)
        else:
            out_csv_path = Path(f"scan_start_of_day_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_res.to_csv(out_csv_path, index=False)
        print(f"\nSaved results to: {out_csv_path}")

    return df_res, df_err, out_csv_path
