"""
Analysis helpers for the weekly 1.5% target strategy.

This module is intentionally analysis-only. It builds clean OHLCV tables,
forward target/stop labels, simple technical features, sector summaries, and
correlation matrices for the B_Data_Analysis notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:  # Keep analysis runnable with the current requirements.txt.
    sns = None


TARGET_RETURN = 0.015
STOP_LOSS = -0.015
HORIZON_SESSIONS = 5

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
QUALITY_COLUMNS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

FEATURE_COLUMNS = [
    "ret_5d",
    "ret_20d",
    "close_above_sma_20",
    "close_above_sma_50",
    "dist_prev_close_sma_20",
    "dist_open_sma_20",
    "dist_open_ema_20",
    "sma_20_slope_5d",
    "rsi_14",
    "macd_hist",
    "bb_width_20",
    "bb_position_20",
    "volume_change_5d",
    "relative_volume_20",
    "dollar_volume_log",
    "realized_vol_20",
    "atr14_norm",
    "gap_open_prev_close",
    "gap_up",
    "gap_down",
    "prev_close_above_prior_20d_high",
    "open_above_prior_20d_high",
    "dist_open_prior_20d_high",
    "prev_close_below_prior_20d_low",
    "dist_open_prior_20d_low",
]

OUTCOME_COLUMNS = [
    "target_hit_5d",
    "target_before_stop_5d",
    "max_forward_return_5d",
    "min_forward_drawdown_5d",
    "close_return_5d",
    "days_to_target",
]


def resolve_project_root(start: Path | None = None) -> Path:
    """Find NN_Trading_project from common notebook or repo working dirs."""
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "A_Data_Gathering").exists() and (candidate / "B_Data_Analysis").exists():
            return candidate
        nested = candidate / "NN_Trading_project"
        if (nested / "A_Data_Gathering").exists() and (nested / "B_Data_Analysis").exists():
            return nested
    raise FileNotFoundError("Could not find NN_Trading_project from current working directory.")


def default_paths(project_root: Path | None = None) -> dict[str, Path]:
    """Return common project paths used by the analysis notebook."""
    root = resolve_project_root() if project_root is None else Path(project_root)
    analysis_dir = root / "B_Data_Analysis"
    return {
        "project_root": root,
        "dataset_dir": root / "A_Data_Gathering" / "dataset" / "stocks",
        "analysis_dir": analysis_dir,
        "sector_map": analysis_dir / "sector_map.json",
        "output_dir": analysis_dir / "outputs",
    }


def load_stock_data(dataset_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all ticker CSVs and normalize Date/OHLCV dtypes.

    The yfinance files in this project contain an extra ticker row after the
    header. That row is removed when present.
    """
    if dataset_dir is None:
        dataset_dir = default_paths()["dataset_dir"]
    dataset_dir = Path(dataset_dir)
    stock_data: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(dataset_dir.glob("*.csv")):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        first_date = str(df.iloc[0].get("Date", "")).strip().lower()
        first_values = {str(v).strip().upper() for v in df.iloc[0].tolist()}
        if first_date in {"", "nan", "nat"} or symbol.upper() in first_values:
            df = df.iloc[1:].copy()

        if "AdjClose" in df.columns and "Adj Close" not in df.columns:
            df = df.rename(columns={"AdjClose": "Adj Close"})

        if "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.set_index("Date")
        df.index.name = "Date"
        stock_data[symbol] = df

    return stock_data


def load_sector_map(sector_map_path: Path | str | None = None) -> dict[str, str]:
    """Load the ticker-to-sector mapping."""
    if sector_map_path is None:
        sector_map_path = default_paths()["sector_map"]
    with open(sector_map_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k).upper(): str(v) for k, v in raw.items()}


def sector_mapping_frame(symbols: list[str], sector_map: dict[str, str]) -> pd.DataFrame:
    """Create a display-friendly sector map and mark missing sectors as Other."""
    rows = []
    for symbol in sorted(symbols):
        sector = sector_map.get(symbol.upper(), "Other")
        rows.append(
            {
                "Ticker": symbol,
                "Sector": sector,
                "Mapping_Status": "Mapped" if symbol.upper() in sector_map else "Missing -> Other",
            }
        )
    return pd.DataFrame(rows)


def data_quality_report(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one quality row per ticker."""
    rows = []
    for symbol, df in stock_data.items():
        work = df.reset_index()
        missing = {
            f"Missing_{col.replace(' ', '_')}": int(work[col].isna().sum())
            for col in QUALITY_COLUMNS
            if col in work.columns
        }
        price_frame = work[[col for col in PRICE_COLUMNS if col in work.columns]]

        non_positive_prices = int((price_frame <= 0).any(axis=1).sum()) if not price_frame.empty else 0
        negative_volume = int((work["Volume"] < 0).sum()) if "Volume" in work.columns else 0
        high_lt_low = int((work["High"] < work["Low"]).sum()) if {"High", "Low"}.issubset(work.columns) else 0
        open_outside_range = (
            int(((work["Open"] > work["High"]) | (work["Open"] < work["Low"])).sum())
            if {"Open", "High", "Low"}.issubset(work.columns)
            else 0
        )
        close_outside_range = (
            int(((work["Close"] > work["High"]) | (work["Close"] < work["Low"])).sum())
            if {"Close", "High", "Low"}.issubset(work.columns)
            else 0
        )

        issue_count = (
            sum(missing.values())
            + int(df.index.duplicated().sum())
            + non_positive_prices
            + negative_volume
            + high_lt_low
            + open_outside_range
            + close_outside_range
        )

        row = {
            "Ticker": symbol,
            "Rows": int(len(df)),
            "Start": df.index.min().date() if len(df) else pd.NaT,
            "End": df.index.max().date() if len(df) else pd.NaT,
            "Duplicate_Dates": int(df.index.duplicated().sum()),
            "Non_Positive_Prices": non_positive_prices,
            "Negative_Volume": negative_volume,
            "High_Less_Than_Low": high_lt_low,
            "Open_Outside_Range": open_outside_range,
            "Close_Outside_Range": close_outside_range,
            "Issue_Count": int(issue_count),
        }
        row.update(missing)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["Issue_Count", "Ticker"], ascending=[False, True])


def compute_forward_outcomes(
    df: pd.DataFrame,
    target_return: float = TARGET_RETURN,
    stop_loss: float = STOP_LOSS,
    horizon_sessions: int = HORIZON_SESSIONS,
) -> pd.DataFrame:
    """Compute future target/stop outcomes for each possible entry row."""
    work = df[["Open", "High", "Low", "Close"]].astype(float).reset_index()
    n_rows = len(work)

    out = pd.DataFrame({"Date": work["Date"]})
    for col in OUTCOME_COLUMNS:
        out[col] = np.nan
    out["valid_forward_window"] = False

    open_ = work["Open"].to_numpy(dtype=float)
    high = work["High"].to_numpy(dtype=float)
    low = work["Low"].to_numpy(dtype=float)
    close = work["Close"].to_numpy(dtype=float)

    last_start = n_rows - int(horizon_sessions)
    if last_start < 0:
        return out

    for start in range(last_start + 1):
        entry = open_[start]
        if not np.isfinite(entry) or entry <= 0:
            continue

        end = start + int(horizon_sessions)
        high_window = high[start:end]
        low_window = low[start:end]
        close_end = close[end - 1]
        target_price = entry * (1.0 + target_return)
        stop_price = entry * (1.0 + stop_loss)

        out.loc[start, "valid_forward_window"] = True
        out.loc[start, "target_hit_5d"] = float(np.nanmax(high_window) >= target_price)
        out.loc[start, "max_forward_return_5d"] = (np.nanmax(high_window) / entry) - 1.0
        out.loc[start, "min_forward_drawdown_5d"] = (np.nanmin(low_window) / entry) - 1.0
        out.loc[start, "close_return_5d"] = (close_end / entry) - 1.0 if np.isfinite(close_end) else np.nan

        target_before_stop = 0.0
        days_to_target = np.nan
        for offset in range(int(horizon_sessions)):
            hit_target = np.isfinite(high_window[offset]) and high_window[offset] >= target_price
            hit_stop = np.isfinite(low_window[offset]) and low_window[offset] <= stop_price

            if hit_target and np.isnan(days_to_target):
                days_to_target = float(offset)

            if hit_target and hit_stop:
                target_before_stop = 0.0
                break
            if hit_stop:
                target_before_stop = 0.0
                break
            if hit_target:
                target_before_stop = 1.0
                break

        out.loc[start, "target_before_stop_5d"] = target_before_stop
        out.loc[start, "days_to_target"] = days_to_target

    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - macd_signal


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _as_float_flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
    return pd.Series(np.where(valid, condition.astype(float), np.nan), index=condition.index)


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal features available by the entry open."""
    work = df.copy()
    close = work["Close"].astype(float)
    high = work["High"].astype(float)
    low = work["Low"].astype(float)
    open_ = work["Open"].astype(float)
    volume = work["Volume"].astype(float)

    prev_close = close.shift(1)
    prev_volume = volume.shift(1)
    logret = np.log(close / close.shift(1))

    features = pd.DataFrame(index=work.index)
    features["ret_5d"] = (prev_close / prev_close.shift(5)) - 1.0
    features["ret_20d"] = (prev_close / prev_close.shift(20)) - 1.0

    sma_5 = close.rolling(5, min_periods=5).mean().shift(1)
    sma_20 = close.rolling(20, min_periods=20).mean().shift(1)
    sma_50 = close.rolling(50, min_periods=50).mean().shift(1)
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean().shift(1)

    features["sma_5"] = sma_5
    features["sma_20"] = sma_20
    features["sma_50"] = sma_50
    features["ema_20"] = ema_20
    features["close_above_sma_20"] = _as_float_flag(prev_close > sma_20, sma_20.notna())
    features["close_above_sma_50"] = _as_float_flag(prev_close > sma_50, sma_50.notna())
    features["dist_prev_close_sma_20"] = (prev_close / sma_20) - 1.0
    features["dist_open_sma_20"] = (open_ / sma_20) - 1.0
    features["dist_open_ema_20"] = (open_ / ema_20) - 1.0
    features["sma_20_slope_5d"] = (sma_20 / sma_20.shift(5)) - 1.0

    features["rsi_14"] = _rsi(close, 14).shift(1)
    features["macd_hist"] = _macd_hist(close).shift(1)

    bb_mid = close.rolling(20, min_periods=20).mean().shift(1)
    bb_std = close.rolling(20, min_periods=20).std().shift(1)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bb_range = (bb_upper - bb_lower).replace(0.0, np.nan)
    features["bb_width_20"] = bb_range / bb_mid
    features["bb_position_20"] = (prev_close - bb_lower) / bb_range

    volume_ma_20 = prev_volume.rolling(20, min_periods=20).mean()
    features["volume_change_5d"] = (prev_volume / prev_volume.shift(5)) - 1.0
    features["relative_volume_20"] = prev_volume / volume_ma_20
    features["dollar_volume_log"] = np.log1p(prev_close * prev_volume)

    features["realized_vol_20"] = logret.shift(1).rolling(20, min_periods=20).std()
    true_range = _true_range(high, low, close)
    atr_14 = true_range.shift(1).rolling(14, min_periods=14).mean()
    features["atr14_norm"] = atr_14 / prev_close

    features["gap_open_prev_close"] = (open_ / prev_close) - 1.0
    features["gap_up"] = _as_float_flag(features["gap_open_prev_close"] > 0, prev_close.notna())
    features["gap_down"] = _as_float_flag(features["gap_open_prev_close"] < 0, prev_close.notna())

    prior_20d_high = high.shift(2).rolling(20, min_periods=20).max()
    prior_20d_low = low.shift(2).rolling(20, min_periods=20).min()
    features["prior_20d_high"] = prior_20d_high
    features["prior_20d_low"] = prior_20d_low
    features["prev_close_above_prior_20d_high"] = _as_float_flag(
        prev_close > prior_20d_high,
        prior_20d_high.notna(),
    )
    features["open_above_prior_20d_high"] = _as_float_flag(
        open_ > prior_20d_high,
        prior_20d_high.notna(),
    )
    features["dist_open_prior_20d_high"] = (open_ / prior_20d_high) - 1.0
    features["prev_close_below_prior_20d_low"] = _as_float_flag(
        prev_close < prior_20d_low,
        prior_20d_low.notna(),
    )
    features["dist_open_prior_20d_low"] = (open_ / prior_20d_low) - 1.0

    return features.replace([np.inf, -np.inf], np.nan)


def build_analysis_table(
    stock_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    target_return: float = TARGET_RETURN,
    stop_loss: float = STOP_LOSS,
    horizon_sessions: int = HORIZON_SESSIONS,
) -> pd.DataFrame:
    """Build one analysis table with raw fields, features, and labels."""
    frames = []
    for symbol, df in stock_data.items():
        if not set(REQUIRED_COLUMNS[1:]).issubset(df.columns):
            continue

        base = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        features = add_analysis_features(df).reset_index(drop=True)
        outcomes = compute_forward_outcomes(df, target_return, stop_loss, horizon_sessions).drop(columns=["Date"])
        joined = pd.concat([base, features, outcomes.reset_index(drop=True)], axis=1)
        joined.insert(1, "Ticker", symbol)
        joined.insert(2, "Sector", sector_map.get(symbol.upper(), "Other"))
        joined["valid_backward_window"] = joined[FEATURE_COLUMNS].notna().all(axis=1)
        joined["valid_analysis_window"] = joined["valid_forward_window"] & joined["valid_backward_window"]
        frames.append(joined)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0, ignore_index=True)


def max_drawdown(close: pd.Series) -> float:
    """Calculate max drawdown from a close-price series."""
    clean = close.dropna().astype(float)
    if clean.empty:
        return np.nan
    running_peak = clean.cummax()
    drawdown = (clean / running_peak) - 1.0
    return float(drawdown.min())


def ticker_return_risk_summary(
    stock_data: dict[str, pd.DataFrame],
    analysis_table: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Simple ticker-level weekly return, risk, and label summary."""
    outcome_group = analysis_table[analysis_table["valid_forward_window"]].groupby("Ticker")
    backward_group = analysis_table[analysis_table["valid_backward_window"]].groupby("Ticker") if "valid_backward_window" in analysis_table.columns else None
    analysis_group = analysis_table[analysis_table["valid_analysis_window"]].groupby("Ticker") if "valid_analysis_window" in analysis_table.columns else None
    rows = []
    for symbol, df in stock_data.items():
        close = df["Close"].dropna().astype(float)
        weekly_close = close.resample("W-FRI").last().dropna()
        weekly_return = weekly_close.pct_change().dropna()
        outcomes = outcome_group.get_group(symbol) if symbol in outcome_group.groups else pd.DataFrame()
        backward_rows = (
            len(backward_group.get_group(symbol))
            if backward_group is not None and symbol in backward_group.groups
            else np.nan
        )
        analysis_rows = (
            len(analysis_group.get_group(symbol))
            if analysis_group is not None and symbol in analysis_group.groups
            else np.nan
        )

        rows.append(
            {
                "Ticker": symbol,
                "Sector": sector_map.get(symbol.upper(), "Other"),
                "Rows": int(len(df)),
                "Valid_Forward_Rows": int(len(outcomes)),
                "Valid_Backward_Rows": backward_rows,
                "Valid_Analysis_Rows": analysis_rows,
                "Total_Return": (close.iloc[-1] / close.iloc[0]) - 1.0 if len(close) > 1 else np.nan,
                "Mean_Weekly_Return": weekly_return.mean(),
                "Weekly_Volatility": weekly_return.std(),
                "Max_Drawdown": max_drawdown(close),
                "Positive_Week_Rate": (weekly_return > 0).mean() if len(weekly_return) else np.nan,
                "Target_Hit_5d_Rate": outcomes["target_hit_5d"].mean() if not outcomes.empty else np.nan,
                "Target_Before_Stop_5d_Rate": outcomes["target_before_stop_5d"].mean() if not outcomes.empty else np.nan,
                "Median_Max_Forward_Return_5d": outcomes["max_forward_return_5d"].median() if not outcomes.empty else np.nan,
                "Median_Min_Forward_Drawdown_5d": outcomes["min_forward_drawdown_5d"].median() if not outcomes.empty else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("Target_Before_Stop_5d_Rate", ascending=False)


def sector_return_risk_summary(ticker_summary: pd.DataFrame, analysis_table: pd.DataFrame) -> pd.DataFrame:
    """Simple sector-level return, risk, and label summary."""
    by_sector = ticker_summary.groupby("Sector").agg(
        Tickers=("Ticker", "nunique"),
        Avg_Total_Return=("Total_Return", "mean"),
        Mean_Weekly_Return=("Mean_Weekly_Return", "mean"),
        Weekly_Volatility=("Weekly_Volatility", "mean"),
        Max_Drawdown=("Max_Drawdown", "mean"),
        Positive_Week_Rate=("Positive_Week_Rate", "mean"),
    )

    valid = analysis_table[analysis_table["valid_forward_window"]].copy()
    label_summary = valid.groupby("Sector").agg(
        Samples=("target_before_stop_5d", "count"),
        Target_Hit_5d_Rate=("target_hit_5d", "mean"),
        Target_Before_Stop_5d_Rate=("target_before_stop_5d", "mean"),
        Median_Forward_Drawdown=("min_forward_drawdown_5d", "median"),
        P10_Forward_Drawdown=("min_forward_drawdown_5d", lambda s: s.quantile(0.10)),
    )

    return by_sector.join(label_summary, how="left").sort_values("Target_Before_Stop_5d_Rate", ascending=False)


def stock_return_matrix(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create an aligned daily return matrix by ticker."""
    returns = {}
    for symbol, df in stock_data.items():
        returns[symbol] = df["Close"].astype(float).pct_change()
    return pd.DataFrame(returns).sort_index()


def sector_return_matrix(stock_data: dict[str, pd.DataFrame], sector_map: dict[str, str]) -> pd.DataFrame:
    """Create an aligned daily return matrix by sector average."""
    returns = stock_return_matrix(stock_data)
    sector_frames = {}
    for sector in sorted(set(sector_map.get(symbol.upper(), "Other") for symbol in returns.columns)):
        members = [symbol for symbol in returns.columns if sector_map.get(symbol.upper(), "Other") == sector]
        if members:
            sector_frames[sector] = returns[members].mean(axis=1)
    return pd.DataFrame(sector_frames).sort_index()


def breakout_summary(analysis_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize simple breakout behavior overall and by sector."""
    valid = analysis_table[analysis_table["valid_forward_window"]].copy()
    valid["Open_Breakout_20d"] = valid["open_above_prior_20d_high"] == 1.0
    valid["Prev_Close_Breakdown_20d"] = valid["prev_close_below_prior_20d_low"] == 1.0

    by_flag = valid.groupby("Open_Breakout_20d").agg(
        Samples=("target_before_stop_5d", "count"),
        Target_Before_Stop_5d_Rate=("target_before_stop_5d", "mean"),
        Target_Hit_5d_Rate=("target_hit_5d", "mean"),
        Median_Forward_Drawdown=("min_forward_drawdown_5d", "median"),
    )

    by_sector = valid.groupby("Sector").agg(
        Samples=("target_before_stop_5d", "count"),
        Breakout_Days=("Open_Breakout_20d", "sum"),
        Breakdown_Days=("Prev_Close_Breakdown_20d", "sum"),
        Breakout_Rate=("Open_Breakout_20d", "mean"),
        Target_Before_Stop_5d_Rate=("target_before_stop_5d", "mean"),
    )
    return by_flag, by_sector.sort_values("Breakout_Rate", ascending=False)


def feature_label_correlations(
    analysis_table: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate feature correlations against label and forward outcome columns."""
    if features is None:
        features = FEATURE_COLUMNS
    if targets is None:
        targets = [
            "target_before_stop_5d",
            "target_hit_5d",
            "max_forward_return_5d",
            "min_forward_drawdown_5d",
            "close_return_5d",
        ]

    valid_features = [col for col in features if col in analysis_table.columns]
    valid_targets = [col for col in targets if col in analysis_table.columns]
    source = analysis_table
    if "valid_analysis_window" in source.columns:
        source = source[source["valid_analysis_window"]].copy()
    numeric = source[valid_features + valid_targets].apply(pd.to_numeric, errors="coerce")

    rows = {}
    for target in valid_targets:
        rows[target] = numeric[valid_features].corrwith(numeric[target])
    corr = pd.DataFrame(rows)

    if "target_before_stop_5d" in corr.columns:
        corr = corr.reindex(corr["target_before_stop_5d"].abs().sort_values(ascending=False).index)
    return corr


def feature_correlation_matrix(analysis_table: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Calculate feature-to-feature correlations."""
    if features is None:
        features = FEATURE_COLUMNS
    valid_features = [col for col in features if col in analysis_table.columns]
    source = analysis_table
    if "valid_analysis_window" in source.columns:
        source = source[source["valid_analysis_window"]].copy()
    return source[valid_features].apply(pd.to_numeric, errors="coerce").corr()


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path | str,
    title: str,
    figsize: tuple[int, int] = (12, 8),
    annot: bool = False,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
    center: float | None = 0.0,
) -> Path:
    """Save a heatmap and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, cmap=cmap, center=center, annot=annot, fmt=fmt, linewidths=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return path

    fig, ax = plt.subplots(figsize=figsize)
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if center == 0.0 and np.isfinite(values).any() else None
    vmin = -vmax if vmax is not None else None
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    if annot:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                val = values[row, col]
                if np.isfinite(val):
                    ax.text(col, row, format(val, fmt), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_bar_chart(
    data: pd.Series | pd.DataFrame,
    path: Path | str,
    title: str,
    ylabel: str = "",
    xlabel: str = "",
    horizontal: bool = False,
    figsize: tuple[int, int] = (12, 7),
    color: str | list[str] = "steelblue",
    sort: bool = False,
) -> Path:
    """Save a simple bar chart from a Series or one-column DataFrame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("save_bar_chart expects a Series or one-column DataFrame.")
        series = data.iloc[:, 0]
    else:
        series = data

    series = series.dropna()
    if sort:
        series = series.sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(series.index.astype(str), series.to_numpy(dtype=float), color=color)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(series.index.astype(str), series.to_numpy(dtype=float), color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pie_chart(
    data: pd.Series | dict,
    path: Path | str,
    title: str,
    figsize: tuple[int, int] = (8, 7),
    startangle: int = 90,
) -> Path:
    """Save a pie chart for small composition-style summaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    series = pd.Series(data, dtype=float).dropna()
    series = series[series > 0]
    if series.empty:
        raise ValueError("Pie chart needs at least one positive value.")

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        series.to_numpy(dtype=float),
        labels=series.index.astype(str),
        autopct="%1.1f%%",
        startangle=startangle,
        pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_grouped_bar_chart(
    data: pd.DataFrame,
    path: Path | str,
    title: str,
    columns: list[str],
    ylabel: str = "",
    xlabel: str = "",
    figsize: tuple[int, int] = (12, 7),
) -> Path:
    """Save a grouped bar chart for selected numeric DataFrame columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = data[columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")

    fig, ax = plt.subplots(figsize=figsize)
    plot_data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_grouped_scatter_chart(
    data: pd.DataFrame,
    path: Path | str,
    x: str,
    y: str,
    group_col: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    label_col: str | None = None,
    annotate_top_n: int = 0,
    annotate_by: str | None = None,
    figsize: tuple[int, int] = (11, 8),
) -> Path:
    """Save a grouped scatter chart for risk/reward maps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = [x, y, group_col]
    plot_data = data.dropna(subset=required).copy()
    if plot_data.empty:
        raise ValueError("No valid rows supplied for grouped scatter chart.")

    fig, ax = plt.subplots(figsize=figsize)
    for group, group_df in plot_data.groupby(group_col):
        ax.scatter(group_df[x], group_df[y], alpha=0.78, s=52, label=str(group))

    if label_col is not None and label_col in plot_data.columns and annotate_top_n > 0:
        sort_col = annotate_by if annotate_by in plot_data.columns else y
        annotations = plot_data.sort_values(sort_col, ascending=False).head(annotate_top_n)
        for _, row in annotations.iterrows():
            ax.annotate(str(row[label_col]), (row[x], row[y]), fontsize=8, alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.25)
    ax.legend(title=group_col, fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_scatter_chart(
    data: pd.DataFrame,
    path: Path | str,
    x: str,
    y: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    label_col: str | None = None,
    figsize: tuple[int, int] = (10, 7),
) -> Path:
    """Save a scatter chart, optionally annotating points with a label column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = data.dropna(subset=[x, y]).copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(plot_data[x], plot_data[y], alpha=0.75)
    if label_col is not None and label_col in plot_data.columns:
        for _, row in plot_data.iterrows():
            ax.annotate(str(row[label_col]), (row[x], row[y]), fontsize=8, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_histogram_grid(
    data: pd.DataFrame,
    path: Path | str,
    columns: list[str],
    title: str,
    bins: int = 40,
    figsize: tuple[int, int] = (14, 10),
) -> Path:
    """Save a grid of histograms for selected numeric columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_columns = [col for col in columns if col in data.columns]
    if not valid_columns:
        raise ValueError("No valid columns supplied for histogram grid.")

    ncols = 2
    nrows = int(np.ceil(len(valid_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, col in zip(axes.ravel(), valid_columns):
        values = pd.to_numeric(data[col], errors="coerce").dropna()
        ax.hist(values, bins=bins, color="steelblue", alpha=0.8)
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(valid_columns):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_moving_average_examples(
    stock_data: dict[str, pd.DataFrame],
    output_path: Path | str,
    symbols: list[str],
) -> Path:
    """Save simple close/SMA/EMA example charts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = [symbol for symbol in symbols if symbol in stock_data][:6]
    if not symbols:
        raise ValueError("No valid symbols supplied for moving-average examples.")

    ncols = 2
    nrows = int(np.ceil(len(symbols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)

    for ax, symbol in zip(axes.ravel(), symbols):
        df = stock_data[symbol]
        close = df["Close"].astype(float)
        ax.plot(close.index, close, label="Close", linewidth=1.0)
        ax.plot(close.rolling(20).mean().index, close.rolling(20).mean(), label="SMA 20", linewidth=0.9)
        ax.plot(close.rolling(50).mean().index, close.rolling(50).mean(), label="SMA 50", linewidth=0.9)
        ax.plot(close.ewm(span=20, adjust=False).mean().index, close.ewm(span=20, adjust=False).mean(), label="EMA 20", linewidth=0.9)
        ax.set_title(symbol)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=35)

    for ax in axes.ravel()[len(symbols):]:
        ax.axis("off")

    fig.suptitle("Simple Moving Average Trend View", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_breakout_examples(
    analysis_table: pd.DataFrame,
    output_path: Path | str,
    symbols: list[str],
) -> Path:
    """Save simple recent high/low breakout example charts from the analysis table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = [symbol for symbol in symbols if symbol in set(analysis_table["Ticker"])][:6]
    if not symbols:
        raise ValueError("No valid symbols supplied for breakout examples.")

    ncols = 2
    nrows = int(np.ceil(len(symbols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)

    for ax, symbol in zip(axes.ravel(), symbols):
        df = analysis_table[analysis_table["Ticker"] == symbol].copy()
        df = df.sort_values("Date")
        ax.plot(df["Date"], df["Close"], label="Close", color="black", linewidth=1.0)
        ax.plot(df["Date"], df["prior_20d_high"], label="Prior 20d High", color="green", linestyle="--", linewidth=0.8)
        ax.plot(df["Date"], df["prior_20d_low"], label="Prior 20d Low", color="red", linestyle="--", linewidth=0.8)

        breakout = df[df["open_above_prior_20d_high"] == 1.0]
        breakdown = df[df["prev_close_below_prior_20d_low"] == 1.0]
        ax.scatter(breakout["Date"], breakout["Open"], marker="^", s=30, color="green", label="Open breakout")
        ax.scatter(breakdown["Date"], breakdown["Close"], marker="v", s=30, color="red", label="Breakdown")
        ax.set_title(symbol)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=35)

    for ax in axes.ravel()[len(symbols):]:
        ax.axis("off")

    fig.suptitle("Simple Breakout Pattern Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
