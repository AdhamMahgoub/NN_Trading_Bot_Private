"""
feature_builder.py
------------------
Feature engineering and cache utilities.

Public API
----------
featurize_one_csv(csv_path, window, horizon_bars, profit_threshold,
                  stop_loss, exclude_start, exclude_end)
    -> (X, y, dates) | None

compute_engineered_features(df, window)
    -> pd.DataFrame   (shared feature builder used by training and inference)

precompute_and_cache(files, window, cache_dir, scaler_path, index_path,
                     horizon_bars, train_end_date, val_end_date,
                     profit_threshold, stop_loss, exclude_start, exclude_end)
    -> (scaler, index)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from helpers.feature.ta_indicators import add_ta_features, atr

# ──────────────────────────────────────────────────────────────────────────────
# Feature columns used by the model (order matters — kept stable here)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "open_rel_prevclose", "high_rel_prevclose", "low_rel_prevclose",
    "gap_prevclose_to_open",
    "logret_1", "ret_co", "range_hl",
    "logret_sum_3", "logret_sum_6", "logret_sum_12",
    "z_logret_1_20",
    "body_size", "upper_wick_norm", "lower_wick_norm", "body_to_range", "clv",
    "atr14_norm", "atr14_impulse_60",
    "gk_vol_20",
    "rv_10", "rv_20", "rv_40", "rv_10_vol_20",
    "range_expansion_z",
    "range_pos_10", "range_pos_20", "z_close_20",
    "dist_to_HH_40", "dist_to_LL_20",
    "dollar_vol_log", "vol_rel_10", "vol_rel_20", "vol_rel_40", "pv_agree_20",
    "obv", "obv_change_20",
    "macd_hist", "rsi_14", "adx_14", "cci_20", "stoch_k_14", "stoch_d_14",
    "mom_vol_6_20", "meanrev_vol_20",
]


# ──────────────────────────────────────────────────────────────────────────────
# Shared feature computation (used by both training and inference)
# ──────────────────────────────────────────────────────────────────────────────

def compute_engineered_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute all engineered features + lagged windows from a prepared OHLCV DataFrame.

    Expects columns: Date, Open, High, Low, Close, Volume (already cleaned/numeric).
    Returns a DataFrame with Date + all feature columns (including lags), rows with
    NaN dropped.  Does NOT compute labels — caller handles that separately.
    """
    df = df.copy()

    # ── Previous-day / current-day raw features ─────────────────────────────
    df["prev_Open"]   = df["Open"].shift(1)
    df["prev_High"]   = df["High"].shift(1)
    df["prev_Low"]    = df["Low"].shift(1)
    df["prev_Close"]  = df["Close"].shift(1)
    df["prev_Volume"] = df["Volume"].shift(1)
    df["today_Open"]  = df["Open"]

    pc = df["prev_Close"]
    denom_close = pc.replace(0.0, np.nan)

    df["logret_1"]              = np.log(pc / pc.shift(1))
    df["ret_co"]                = (pc / df["prev_Open"]) - 1.0
    df["range_hl"]              = (df["prev_High"] - df["prev_Low"]) / denom_close
    df["gap_prevclose_to_open"] = (df["today_Open"] / denom_close) - 1.0

    close_t2 = pc.shift(1)
    df["open_rel_prevclose"]  = (df["prev_Open"]  / close_t2) - 1.0
    df["high_rel_prevclose"]  = (df["prev_High"]  / close_t2) - 1.0
    df["low_rel_prevclose"]   = (df["prev_Low"]   / close_t2) - 1.0

    # Candle anatomy
    body       = pc - df["prev_Open"]
    upper_raw  = df["prev_High"] - pd.concat([df["prev_Open"], pc], axis=1).max(axis=1)
    lower_raw  = pd.concat([df["prev_Open"], pc], axis=1).min(axis=1) - df["prev_Low"]
    range_raw  = (df["prev_High"] - df["prev_Low"]).replace(0.0, np.nan)

    df["body_size"]       = body / denom_close
    df["upper_wick_norm"] = upper_raw / denom_close
    df["lower_wick_norm"] = lower_raw / denom_close
    df["body_to_range"]   = body.abs() / range_raw
    df["clv"]             = ((pc - df["prev_Low"]) - (df["prev_High"] - pc)) / range_raw

    # ATR
    df["atr14"]            = atr(df["prev_High"], df["prev_Low"], pc, period=14)
    df["atr14_norm"]       = df["atr14"] / denom_close
    df["atr14_impulse_60"] = df["atr14_norm"] / df["atr14_norm"].rolling(60, min_periods=60).mean()

    # Returns rolling (logret_mean_* not in FEATURE_COLS — kept for experimentation)
    for win in (3, 6, 12):
        roll = df["logret_1"].rolling(win, min_periods=win).sum()
        df[f"logret_sum_{win}"]  = roll
        df[f"logret_mean_{win}"] = roll / float(win)

    # Price slope (slope_close_* not in FEATURE_COLS — kept for experimentation)
    for win in (3, 6, 12):
        df[f"slope_close_{win}"] = (pc - pc.shift(win)) / float(win)

    # Distance to high/low (dist_to_HH_20, dist_to_LL_40 not in FEATURE_COLS — kept for experimentation)
    for win in (20, 40):
        roll_max = pc.rolling(win, min_periods=win).max()
        roll_min = pc.rolling(win, min_periods=win).min()
        df[f"dist_to_HH_{win}"] = (pc / roll_max) - 1.0
        df[f"dist_to_LL_{win}"] = (pc / roll_min) - 1.0

    # Realised vol
    for win in (10, 20, 40):
        df[f"rv_{win}"] = df["logret_1"].rolling(win, min_periods=win).std()
    df["rv_10_vol_20"] = df["rv_10"].rolling(20, min_periods=20).std()

    re_mean = df["range_hl"].rolling(20, min_periods=20).mean()
    re_std  = df["range_hl"].rolling(20, min_periods=20).std()
    df["range_expansion_z"] = (df["range_hl"] - re_mean) / re_std

    # Range position
    for win in (10, 20):
        roll_min_l = df["prev_Low"].rolling(win, min_periods=win).min()
        roll_max_h = df["prev_High"].rolling(win, min_periods=win).max()
        denom_r    = (roll_max_h - roll_min_l).replace(0.0, np.nan)
        df[f"range_pos_{win}"] = (pc - roll_min_l) / denom_r

    # Z-score
    df["z_close_20"]    = (pc - pc.rolling(20, min_periods=20).mean()) / pc.rolling(20, min_periods=20).std()
    df["z_logret_1_20"] = (df["logret_1"] - df["logret_1"].rolling(20, min_periods=20).mean()) / df["logret_1"].rolling(20, min_periods=20).std()

    # Parkinson / GK vol (parkinson_20 not in FEATURE_COLS — kept for experimentation)
    hl_ratio   = (df["prev_High"] / df["prev_Low"]).replace({0.0: np.nan})
    log_hl     = np.log(hl_ratio)
    log_co     = np.log((pc / df["prev_Open"]).replace({0.0: np.nan}))
    df["parkinson_20"] = ((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20, min_periods=20).mean()) ** 0.5
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df["gk_vol_20"] = (gk_var.rolling(20, min_periods=20).mean()) ** 0.5

    # Volume
    df["dollar_vol_log"] = np.log1p(pc * df["prev_Volume"])
    for win in (10, 20, 40):
        vol_ma = df["prev_Volume"].rolling(win, min_periods=win).mean()
        df[f"vol_rel_{win}"] = df["prev_Volume"] / vol_ma
    df["pv_agree_20"] = df["logret_1"] * df["vol_rel_20"]

    sign_close = np.sign(pc.diff().fillna(0.0))
    df["obv"]            = (sign_close * df["prev_Volume"]).cumsum()
    df["obv_change_20"]  = df["obv"] - df["obv"].shift(20)

    # TA indicators (computed on previous bar's OHLC)
    df_prev = pd.DataFrame({
        "High": df["prev_High"], "Low": df["prev_Low"],
        "Close": pc,             "Open": df["prev_Open"],
    })
    df_prev_ta = add_ta_features(df_prev)
    for col in df_prev_ta.columns:
        if col not in ("High", "Low", "Close", "Open"):
            df[col] = df_prev_ta[col]

    # EMA distances and slopes (dist_to_ema_*, ema_slope_* not in FEATURE_COLS — kept for experimentation)
    for span in (10, 20, 50):
        ec = f"ema_{span}"
        df[f"dist_to_ema_{span}"] = (pc / df[ec]) - 1.0
        df[f"ema_slope_{span}"]   = df[ec] - df[ec].shift(1)

    # Interaction features (trend_vol_20 not in FEATURE_COLS — kept for experimentation)
    df["trend_vol_20"]   = df["dist_to_ema_20"] * df["rv_20"]
    df["mom_vol_6_20"]   = df["logret_sum_6"]   * df["vol_rel_20"]
    df["meanrev_vol_20"] = df["z_close_20"] / (df["rv_20"] + 1e-12)

    # VWAP (vwap_20, dist_to_vwap_20 not in FEATURE_COLS — kept for experimentation)
    tp = (df["prev_High"] + df["prev_Low"] + pc) / 3.0
    vol_sum = df["prev_Volume"].rolling(20, min_periods=20).sum()
    vwap_num = (tp * df["prev_Volume"]).rolling(20, min_periods=20).sum()
    df["vwap_20"]         = (vwap_num / vol_sum).replace([np.inf, -np.inf], np.nan)
    df["dist_to_vwap_20"] = (pc / df["vwap_20"]) - 1.0

    # ── Lagged window ───────────────────────────────────────────────────────
    lagged = [df[FEATURE_COLS].shift(lag).add_suffix(f"_lag_{lag}") for lag in range(1, window)]
    Xdf = pd.concat([df[FEATURE_COLS]] + lagged, axis=1)
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    out = pd.concat([df[["Date"]], Xdf], axis=1)
    out = out.dropna(axis=0).reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Label builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_labels(df: pd.DataFrame, horizon: int, profit_threshold: float, stop_loss: float) -> np.ndarray:
    open_  = df["Open"].astype(np.float32)
    high   = df["High"].astype(np.float32)
    low    = df["Low"].astype(np.float32)

    stop_price   = open_ * (1.0 + stop_loss)
    target_price = open_ * (1.0 + profit_threshold)

    y = np.empty(len(df), dtype=np.float32)
    y[:] = np.nan

    for t in range(len(df)):
        if t + horizon >= len(df):
            continue
        stop_hit   = False
        profit_hit = False

        for k in range(horizon + 1):
            idx = t + k
            if idx >= len(df):
                break
            f_open = open_.iloc[idx]
            f_high = high.iloc[idx]
            f_low  = low.iloc[idx]

            hit_profit = pd.notna(f_high) and f_high >= target_price.iloc[t]
            hit_stop   = pd.notna(f_low)  and f_low  <= stop_price.iloc[t]

            if hit_profit and hit_stop:
                profit_hit = True
                stop_hit   = True
                break
            elif hit_profit:
                profit_hit = True
                break
            elif hit_stop:
                stop_hit = True
                break

        if profit_hit and not stop_hit:
            y[t] = 1.0
        elif pd.notna(open_.iloc[t]):
            y[t] = 0.0

    return y


# ──────────────────────────────────────────────────────────────────────────────
# Core feature builder for a single CSV
# ──────────────────────────────────────────────────────────────────────────────

def featurize_one_csv(
    csv_path: Path,
    window: int,
    horizon_bars: int,
    profit_threshold: float,
    stop_loss: float,
    exclude_start: str,
    exclude_end: str,
):
    """Build (X, y, dates) for one ticker CSV.

    Returns None if the file is invalid or too short.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "AdjClose" in df.columns and "Adj Close" not in df.columns:
        df = df.rename(columns={"AdjClose": "Adj Close"})

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Adj Close"] = (
        pd.to_numeric(df["Adj Close"], errors="coerce")
        if "Adj Close" in df.columns else df["Close"]
    )
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    horizon = int(horizon_bars)
    if len(df) < max(window + horizon + 2, 2):
        return None

    # Exclusion mask — bars in the excluded range (plus their forward look-ahead) are dropped
    excl_s = pd.to_datetime(exclude_start)
    excl_e = pd.to_datetime(exclude_end)
    in_excl     = (df["Date"] >= excl_s) & (df["Date"] <= excl_e)
    excl_future = in_excl[::-1].rolling(horizon + 1, min_periods=1).max()[::-1].astype(bool)
    df["__excl__"] = excl_future

    # Labels
    df["y"] = _build_labels(df, horizon, profit_threshold, stop_loss)

    # ── Previous-day / current-day raw features ──────────────────────────────
    df["prev_Open"]   = df["Open"].shift(1)
    df["prev_High"]   = df["High"].shift(1)
    df["prev_Low"]    = df["Low"].shift(1)
    df["prev_Close"]  = df["Close"].shift(1)
    df["prev_Volume"] = df["Volume"].shift(1)
    df["today_Open"]  = df["Open"]

    pc = df["prev_Close"]
    denom_close = pc.replace(0.0, np.nan)

    df["logret_1"]            = np.log(pc / pc.shift(1))
    df["ret_co"]              = (pc / df["prev_Open"]) - 1.0
    df["range_hl"]            = (df["prev_High"] - df["prev_Low"]) / denom_close
    df["gap_prevclose_to_open"] = (df["today_Open"] / denom_close) - 1.0

    close_t2 = pc.shift(1)
    df["open_rel_prevclose"]  = (df["prev_Open"]  / close_t2) - 1.0
    df["high_rel_prevclose"]  = (df["prev_High"]  / close_t2) - 1.0
    df["low_rel_prevclose"]   = (df["prev_Low"]   / close_t2) - 1.0

    # Candle anatomy
    body       = pc - df["prev_Open"]
    upper_raw  = df["prev_High"] - pd.concat([df["prev_Open"], pc], axis=1).max(axis=1)
    lower_raw  = pd.concat([df["prev_Open"], pc], axis=1).min(axis=1) - df["prev_Low"]
    range_raw  = (df["prev_High"] - df["prev_Low"]).replace(0.0, np.nan)

    df["body_size"]       = body / denom_close
    df["upper_wick_norm"] = upper_raw / denom_close
    df["lower_wick_norm"] = lower_raw / denom_close
    df["body_to_range"]   = body.abs() / range_raw
    df["clv"]             = ((pc - df["prev_Low"]) - (df["prev_High"] - pc)) / range_raw

    # ATR
    df["atr14"]          = atr(df["prev_High"], df["prev_Low"], pc, period=14)
    df["atr14_norm"]     = df["atr14"] / denom_close
    df["atr14_impulse_60"] = df["atr14_norm"] / df["atr14_norm"].rolling(60, min_periods=60).mean()

    # Returns rolling
    for win in (3, 6, 12):
        roll = df["logret_1"].rolling(win, min_periods=win).sum()
        df[f"logret_sum_{win}"]  = roll
        df[f"logret_mean_{win}"] = roll / float(win)

    # Price slope
    for win in (3, 6, 12):
        df[f"slope_close_{win}"] = (pc - pc.shift(win)) / float(win)

    # Distance to high/low
    for win in (20, 40):
        roll_max = pc.rolling(win, min_periods=win).max()
        roll_min = pc.rolling(win, min_periods=win).min()
        df[f"dist_to_HH_{win}"] = (pc / roll_max) - 1.0
        df[f"dist_to_LL_{win}"] = (pc / roll_min) - 1.0

    # Realised vol
    for win in (10, 20, 40):
        df[f"rv_{win}"] = df["logret_1"].rolling(win, min_periods=win).std()
    df["rv_10_vol_20"] = df["rv_10"].rolling(20, min_periods=20).std()

    re_mean = df["range_hl"].rolling(20, min_periods=20).mean()
    re_std  = df["range_hl"].rolling(20, min_periods=20).std()
    df["range_expansion_z"] = (df["range_hl"] - re_mean) / re_std

    # Range position
    for win in (10, 20):
        roll_min_l = df["prev_Low"].rolling(win, min_periods=win).min()
        roll_max_h = df["prev_High"].rolling(win, min_periods=win).max()
        denom_r    = (roll_max_h - roll_min_l).replace(0.0, np.nan)
        df[f"range_pos_{win}"] = (pc - roll_min_l) / denom_r

    # Z-score
    df["z_close_20"]   = (pc - pc.rolling(20, min_periods=20).mean()) / pc.rolling(20, min_periods=20).std()
    df["z_logret_1_20"] = (df["logret_1"] - df["logret_1"].rolling(20, min_periods=20).mean()) / df["logret_1"].rolling(20, min_periods=20).std()

    # Parkinson / GK vol
    hl_ratio   = (df["prev_High"] / df["prev_Low"]).replace({0.0: np.nan})
    log_hl     = np.log(hl_ratio)
    log_co     = np.log((pc / df["prev_Open"]).replace({0.0: np.nan}))
    df["parkinson_20"] = ((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20, min_periods=20).mean()) ** 0.5
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df["gk_vol_20"] = (gk_var.rolling(20, min_periods=20).mean()) ** 0.5

    # Volume
    df["dollar_vol_log"] = np.log1p(pc * df["prev_Volume"])
    for win in (10, 20, 40):
        vol_ma = df["prev_Volume"].rolling(win, min_periods=win).mean()
        df[f"vol_rel_{win}"] = df["prev_Volume"] / vol_ma
    df["pv_agree_20"] = df["logret_1"] * df["vol_rel_20"]

    sign_close = np.sign(pc.diff().fillna(0.0))
    df["obv"]          = (sign_close * df["prev_Volume"]).cumsum()
    df["obv_change_20"] = df["obv"] - df["obv"].shift(20)

    # TA indicators (computed on previous bar's OHLC)
    df_prev = pd.DataFrame({
        "High": df["prev_High"], "Low": df["prev_Low"],
        "Close": pc,             "Open": df["prev_Open"],
    })
    df_prev_ta = add_ta_features(df_prev)
    for col in df_prev_ta.columns:
        if col not in ("High", "Low", "Close", "Open"):
            df[col] = df_prev_ta[col]

    # EMA distances and slopes
    for span in (10, 20, 50):
        ec = f"ema_{span}"
        df[f"dist_to_ema_{span}"] = (pc / df[ec]) - 1.0
        df[f"ema_slope_{span}"]   = df[ec] - df[ec].shift(1)

    # Interaction features
    df["trend_vol_20"]  = df["dist_to_ema_20"] * df["rv_20"]
    df["mom_vol_6_20"]  = df["logret_sum_6"]   * df["vol_rel_20"]
    df["meanrev_vol_20"] = df["z_close_20"] / (df["rv_20"] + 1e-12)

    # VWAP
    tp = (df["prev_High"] + df["prev_Low"] + pc) / 3.0
    vol_sum = df["prev_Volume"].rolling(20, min_periods=20).sum()
    vwap_num = (tp * df["prev_Volume"]).rolling(20, min_periods=20).sum()
    df["vwap_20"]         = (vwap_num / vol_sum).replace([np.inf, -np.inf], np.nan)
    df["dist_to_vwap_20"] = (pc / df["vwap_20"]) - 1.0

    # ── Lagged window ────────────────────────────────────────────────────────
    lagged = [df[FEATURE_COLS].shift(lag).add_suffix(f"_lag_{lag}") for lag in range(1, window)]
    Xdf = pd.concat([df[FEATURE_COLS]] + lagged, axis=1)
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    out = pd.concat([df[["Date"]], Xdf, df[["y", "__excl__"]]], axis=1)
    out = out.dropna(subset=["y"]).dropna(axis=0)
    out = out[~out["__excl__"]].reset_index(drop=True)
    if len(out) < 2:
        return None

    dates = out["Date"].to_numpy(dtype="datetime64[ns]")
    X     = out.drop(columns=["Date", "y", "__excl__"]).to_numpy(dtype=np.float32, copy=True)
    y     = out["y"].to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite BUY labels in {csv_path}")

    return X, y, dates


# ──────────────────────────────────────────────────────────────────────────────
# Cache builder
# ──────────────────────────────────────────────────────────────────────────────

def precompute_and_cache(
    files,
    window: int,
    cache_dir: Path,
    scaler_path: Path,
    index_path: Path,
    horizon_bars: int,
    train_end_date,
    val_end_date,
    profit_threshold: float,
    stop_loss: float,
    exclude_start: str,
    exclude_end: str,
):
    """Featurize all CSVs, fit a per-split-safe StandardScaler, and persist cache.

    Train / Val / Test boundaries are determined by date:
      Train : [0 : train_end) minus embargo of horizon_bars
      Val   : [train_end : val_end) minus embargo of horizon_bars
      Test  : [val_end : N)
    """
    print("[Cache] Precomputing features (leakage-safe splits)...")

    scaler = StandardScaler()
    index  = []
    used = skipped = 0
    embargo = int(horizon_bars)

    train_end = pd.Timestamp(train_end_date).to_datetime64()
    val_end   = pd.Timestamp(val_end_date).to_datetime64()

    for i, f in enumerate(files, start=1):
        print(f"[Cache] ({i}/{len(files)}) {f.name}")

        result = featurize_one_csv(f, window, horizon_bars, profit_threshold, stop_loss, exclude_start, exclude_end)
        if result is None:
            skipped += 1
            continue

        X, y, dates = result
        n = len(X)
        if n < 5:
            skipped += 1
            continue

        # Sort guard
        if not np.all(dates[:-1] <= dates[1:]):
            order = np.argsort(dates)
            dates, X, y = dates[order], X[order], y[order]

        train_end_idx = int(np.searchsorted(dates, train_end, side="right"))
        val_end_idx   = int(np.searchsorted(dates, val_end,   side="right"))

        train_cut  = max(0, train_end_idx - embargo)
        val_start  = train_end_idx
        val_end_cut = max(val_start, val_end_idx - embargo)
        test_start = val_end_idx

        if train_cut < 2 or (val_end_cut - val_start) < 1 or (n - test_start) < 1:
            skipped += 1
            continue

        scaler.partial_fit(X[:train_cut])

        cache_file = cache_dir / f"{f.stem}.npz"
        np.savez_compressed(
            cache_file,
            X=X.astype(np.float32, copy=False),
            y=y.astype(np.float32, copy=False),
            dates=dates.astype("datetime64[ns]"),
        )

        index.append({
            "ticker":     f.stem,
            "cache_file": str(cache_file),
            "train_cut":  int(train_cut),
            "val_start":  int(val_start),
            "val_end":    int(val_end_cut),
            "test_start": int(test_start),
            "total_len":  int(n),
            # debug
            "train_end_idx": int(train_end_idx),
            "val_end_idx":   int(val_end_idx),
            "embargo":       int(embargo),
        })
        used += 1

    if used == 0:
        raise RuntimeError(f"No valid tickers after featurization. skipped={skipped}/{len(files)}")

    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(index_path, "wb") as fh:
        pickle.dump(index, fh)

    print(f"[Cache] Done — used={used}, skipped={skipped}")
    return scaler, index
