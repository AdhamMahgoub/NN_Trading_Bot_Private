"""
ta_indicators.py
----------------
Pure, causal technical-analysis helpers (no external TA library).
All functions operate on pandas Series / DataFrames and return the same type.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Trend / Moving Averages
# ──────────────────────────────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast   = ema(close, fast)
    ema_slow   = ema(close, slow)
    macd_line  = ema_fast - ema_slow
    macd_sig   = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist  = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist


# ──────────────────────────────────────────────
# Momentum / Oscillators
# ──────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    ll    = low.rolling(period, min_periods=period).min()
    hh    = high.rolling(period, min_periods=period).max()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * (close - ll) / denom


def stoch_d(k: pd.Series, smooth: int = 3) -> pd.Series:
    return k.rolling(smooth, min_periods=smooth).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp  = (high + low + close) / 3.0
    sma = tp.rolling(n, min_periods=n).mean()
    md  = tp.rolling(n, min_periods=n).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * (md + 1e-12))


# ──────────────────────────────────────────────
# Volatility
# ──────────────────────────────────────────────

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ──────────────────────────────────────────────
# Trend Strength
# ──────────────────────────────────────────────

def adx_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (adx, plus_di, minus_di)."""
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_smooth       = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_dm_smooth  = pd.Series(plus_dm,  index=high.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    plus_di  = 100.0 * (plus_dm_smooth  / (tr_smooth + 1e-12))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-12))
    dx       = 100.0 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    adx_val  = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return adx_val, plus_di, minus_di


# ──────────────────────────────────────────────
# Composite helper: add all TA columns to a df
# ──────────────────────────────────────────────

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA, MACD, RSI, Stoch, CCI, ADX/DMI columns in-place (returns copy)."""
    out = df.copy()
    c, h, l = out["Close"], out["High"], out["Low"]

    out["ema_10"] = ema(c, 10)
    out["ema_20"] = ema(c, 20)
    out["ema_50"] = ema(c, 50)

    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    out["macd"]        = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"]   = macd_hist

    out["rsi_14"] = rsi(c, 14)

    k = stoch_k(h, l, c, 14)
    out["stoch_k_14"] = k
    out["stoch_d_14"] = stoch_d(k, 3)

    out["cci_20"] = cci(h, l, c, 20)

    adx_val, plus_di, minus_di = adx_dmi(h, l, c, 14)
    out["adx_14"]     = adx_val
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
