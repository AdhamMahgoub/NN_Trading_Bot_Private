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
import xgboost as xgb
import yfinance as yf


def _predict_probs_booster(booster: xgb.Booster, X: np.ndarray, best_ntree: int) -> np.ndarray:
    """Predict probabilities from a native xgboost Booster (version-safe)."""
    dmat = xgb.DMatrix(X)
    try:
        probs = booster.predict(dmat, iteration_range=(0, best_ntree))
    except TypeError:
        probs = booster.predict(dmat, ntree_limit=best_ntree)
    return probs.astype(np.float32)


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    ll = low.rolling(period, min_periods=period).min()
    hh = high.rolling(period, min_periods=period).max()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * (close - ll) / denom


def _stoch_d(stoch_k: pd.Series, smooth: int = 3) -> pd.Series:
    return stoch_k.rolling(smooth, min_periods=smooth).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * (md + 1e-12))


def _adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_smooth = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()

    plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-12))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-12))

    dx = 100.0 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx, plus_di, minus_di


def _add_my_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend/EMA/MACD/RSI/Stoch/CCI/ADX (causal)."""
    out = df.copy()
    c = out["Close"]
    h = out["High"]
    l = out["Low"]

    out["ema_10"] = _ema(c, 10)
    out["ema_20"] = _ema(c, 20)
    out["ema_50"] = _ema(c, 50)

    macd, macd_sig, macd_hist = _macd(c, 12, 26, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    out["rsi_14"] = _rsi(c, 14)

    stoch_k = _stoch_k(h, l, c, 14)
    out["stoch_k_14"] = stoch_k
    out["stoch_d_14"] = _stoch_d(stoch_k, 3)

    out["cci_20"] = _cci(h, l, c, 20)

    adx, plus_di, minus_di = _adx_dmi(h, l, c, 14)
    out["adx_14"] = adx
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


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

    df["prev_Open"] = df["Open"].shift(1)
    df["prev_High"] = df["High"].shift(1)
    df["prev_Low"] = df["Low"].shift(1)
    df["prev_Close"] = df["Close"].shift(1)
    df["prev_Volume"] = df["Volume"].shift(1)
    df["today_Open"] = df["Open"]

    df["logret_1"] = np.log(df["prev_Close"] / df["prev_Close"].shift(1))
    df["ret_co"] = (df["prev_Close"] / df["prev_Open"]) - 1.0
    df["range_hl"] = (df["prev_High"] - df["prev_Low"]) / df["prev_Close"]
    df["gap_prevclose_to_open"] = (df["today_Open"] / df["prev_Close"]) - 1.0

    close_t2 = df["prev_Close"].shift(1)
    df["open_rel_prevclose"] = (df["prev_Open"] / close_t2) - 1.0
    df["high_rel_prevclose"] = (df["prev_High"] / close_t2) - 1.0
    df["low_rel_prevclose"] = (df["prev_Low"] / close_t2) - 1.0

    body = df["prev_Close"] - df["prev_Open"]
    denom_close = df["prev_Close"].replace(0.0, np.nan)
    df["body_size"] = body / denom_close
    upper_raw = df["prev_High"] - pd.concat([df["prev_Open"], df["prev_Close"]], axis=1).max(axis=1)
    lower_raw = pd.concat([df["prev_Open"], df["prev_Close"]], axis=1).min(axis=1) - df["prev_Low"]
    df["upper_wick_norm"] = upper_raw / denom_close
    df["lower_wick_norm"] = lower_raw / denom_close
    range_raw = (df["prev_High"] - df["prev_Low"]).replace(0.0, np.nan)
    df["body_to_range"] = body.abs() / range_raw
    df["clv"] = ((df["prev_Close"] - df["prev_Low"]) - (df["prev_High"] - df["prev_Close"])) / range_raw

    df["atr14"] = _atr(df["prev_High"], df["prev_Low"], df["prev_Close"], period=14)
    df["atr14_norm"] = df["atr14"] / denom_close
    df["atr14_impulse_60"] = df["atr14_norm"] / df["atr14_norm"].rolling(60, min_periods=60).mean()

    for win in (3, 6, 12):
        roll = df["logret_1"].rolling(win, min_periods=win).sum()
        df[f"logret_sum_{win}"] = roll
        df[f"logret_mean_{win}"] = roll / float(win)

    for win in (3, 6, 12):
        df[f"slope_close_{win}"] = (df["prev_Close"] - df["prev_Close"].shift(win)) / float(win)

    roll_max_40 = df["prev_Close"].rolling(40, min_periods=40).max()
    roll_min_20 = df["prev_Close"].rolling(20, min_periods=20).min()
    df["dist_to_HH_40"] = (df["prev_Close"] / roll_max_40) - 1.0
    df["dist_to_LL_20"] = (df["prev_Close"] / roll_min_20) - 1.0

    for win in (10, 20, 40):
        df[f"rv_{win}"] = df["logret_1"].rolling(win, min_periods=win).std()
    df["rv_10_vol_20"] = df["rv_10"].rolling(20, min_periods=20).std()

    re_mean = df["range_hl"].rolling(20, min_periods=20).mean()
    re_std = df["range_hl"].rolling(20, min_periods=20).std()
    df["range_expansion_z"] = (df["range_hl"] - re_mean) / re_std

    for win in (10, 20):
        roll_min_l = df["prev_Low"].rolling(win, min_periods=win).min()
        roll_max_h = df["prev_High"].rolling(win, min_periods=win).max()
        denom_range = (roll_max_h - roll_min_l).replace(0.0, np.nan)
        df[f"range_pos_{win}"] = (df["prev_Close"] - roll_min_l) / denom_range

    close_mean_20 = df["prev_Close"].rolling(20, min_periods=20).mean()
    close_std_20 = df["prev_Close"].rolling(20, min_periods=20).std()
    df["z_close_20"] = (df["prev_Close"] - close_mean_20) / close_std_20

    logret_mean_20 = df["logret_1"].rolling(20, min_periods=20).mean()
    logret_std_20 = df["logret_1"].rolling(20, min_periods=20).std()
    df["z_logret_1_20"] = (df["logret_1"] - logret_mean_20) / logret_std_20

    hl_ratio = (df["prev_High"] / df["prev_Low"]).replace({0.0: np.nan})
    parkinson_bar = np.log(hl_ratio) ** 2
    parkinson_const = 1.0 / (4.0 * np.log(2.0))
    df["parkinson_20"] = (parkinson_const * parkinson_bar.rolling(20, min_periods=20).mean()) ** 0.5

    log_hl = np.log((df["prev_High"] / df["prev_Low"]).replace({0.0: np.nan}))
    log_co = np.log((df["prev_Close"] / df["prev_Open"]).replace({0.0: np.nan}))
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    df["gk_vol_20"] = (gk_var.rolling(20, min_periods=20).mean()) ** 0.5

    df["dollar_vol_log"] = np.log1p(df["prev_Close"] * df["prev_Volume"])
    for win in (10, 20, 40):
        vol_ma = df["prev_Volume"].rolling(win, min_periods=win).mean()
        df[f"vol_rel_{win}"] = df["prev_Volume"] / vol_ma
    df["pv_agree_20"] = df["logret_1"] * df["vol_rel_20"]

    sign_close = np.sign(df["prev_Close"].diff().fillna(0.0))
    df["obv"] = (sign_close * df["prev_Volume"]).cumsum()
    df["obv_change_20"] = df["obv"] - df["obv"].shift(20)

    df_prev = pd.DataFrame(
        {
            "High": df["prev_High"],
            "Low": df["prev_Low"],
            "Close": df["prev_Close"],
            "Open": df["prev_Open"],
        }
    )
    df_prev = _add_my_ta_features(df_prev)
    for col in df_prev.columns:
        if col not in ["High", "Low", "Close", "Open"]:
            df[col] = df_prev[col]

    for span in (10, 20, 50):
        ema_col = f"ema_{span}"
        df[f"dist_to_ema_{span}"] = (df["prev_Close"] / df[ema_col]) - 1.0
        df[f"ema_slope_{span}"] = df[ema_col] - df[ema_col].shift(1)

    df["trend_vol_20"] = df["dist_to_ema_20"] * df["rv_20"]
    df["mom_vol_6_20"] = df["logret_sum_6"] * df["vol_rel_20"]
    df["meanrev_vol_20"] = df["z_close_20"] / (df["rv_20"] + 1e-12)

    tp = (df["prev_High"] + df["prev_Low"] + df["prev_Close"]) / 3.0
    vol_roll_20 = df["prev_Volume"].rolling(20, min_periods=20).sum()
    vwap_num_20 = (tp * df["prev_Volume"]).rolling(20, min_periods=20).sum()
    df["vwap_20"] = (vwap_num_20 / vol_roll_20).replace([np.inf, -np.inf], np.nan)
    df["dist_to_vwap_20"] = (df["prev_Close"] / df["vwap_20"]) - 1.0

    feat_now = [
        "open_rel_prevclose",
        "high_rel_prevclose",
        "low_rel_prevclose",
        "gap_prevclose_to_open",
        "logret_1",
        "ret_co",
        "range_hl",
        "logret_sum_3",
        "logret_sum_6",
        "logret_sum_12",
        "z_logret_1_20",
        "body_size",
        "upper_wick_norm",
        "lower_wick_norm",
        "body_to_range",
        "clv",
        "atr14_norm",
        "atr14_impulse_60",
        "gk_vol_20",
        "rv_10",
        "rv_20",
        "rv_40",
        "rv_10_vol_20",
        "range_expansion_z",
        "range_pos_10",
        "range_pos_20",
        "z_close_20",
        "dist_to_HH_40",
        "dist_to_LL_20",
        "dollar_vol_log",
        "vol_rel_10",
        "vol_rel_20",
        "vol_rel_40",
        "pv_agree_20",
        "obv",
        "obv_change_20",
        "macd_hist",
        "rsi_14",
        "adx_14",
        "cci_20",
        "stoch_k_14",
        "stoch_d_14",
        "mom_vol_6_20",
        "meanrev_vol_20",
    ]

    lagged = [df[feat_now].shift(lag).add_suffix(f"_lag_{lag}") for lag in range(1, window)]
    Xdf = pd.concat([df[feat_now]] + lagged, axis=1)
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    out = pd.concat([df[["Date"]], Xdf], axis=1)
    out = out.dropna(axis=0).reset_index(drop=True)
    return out


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
        prob_buy = float(_predict_probs_booster(booster, Xs, best_ntree)[0])

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
