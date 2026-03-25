"""
╔══════════════════════════════════════════════════════════════════════╗
║              STOCK ANALYSIS ENGINE — indicators.py                  ║
║               Module 2: Technical Indicators Layer                   ║
╚══════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Compute all technical indicators needed for stock analysis:
      - RSI   (Relative Strength Index)      → momentum / exhaustion
      - MACD  (Moving Avg Convergence/Div.)  → trend direction + momentum
      - SMA_20 / SMA_50                      → short & medium-term trend
      - Bollinger Bands                      → volatility + price extremes
      - Volume MA                            → volume context

PHILOSOPHY:
    Each indicator answers ONE specific question about the stock.
    By combining them, we build a complete picture of stock health —
    like a doctor reviewing multiple vitals before making a diagnosis.

DEPENDENCIES:
    pip install pandas numpy pandas-ta
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — RSI  (Relative Strength Index)
# ══════════════════════════════════════════════════════════════════════

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    Formula:
        RS  = Average Gain / Average Loss  (over `period` days)
        RSI = 100 - (100 / (1 + RS))

    Args:
        df     : DataFrame with 'Close' column
        period : Lookback window (default 14 days — standard J. Welles Wilder)

    Returns:
        pd.Series of RSI values (0–100 scale)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why RSI Was Invented
    ──────────────────────────────────────────────
    J. Welles Wilder created RSI in 1978 to answer a fundamental problem:
    "Price has moved a lot — but is that move SUSTAINABLE or EXHAUSTED?"

    RSI measures the SPEED and MAGNITUDE of price changes.
    It doesn't care about direction — it measures MOMENTUM STRENGTH.

    The key thresholds:
      RSI > 70  → OVERBOUGHT: Stock has risen too fast. Sellers may enter.
                  Think: rubber band stretched too far — will snap back.
      RSI < 30  → OVERSOLD:   Stock has fallen too fast. Buyers may step in.
                  Think: a coiled spring — pressure building for a bounce.
      RSI 40–60 → NEUTRAL:    No strong momentum signal.
      RSI 50    → The battleground: above = bulls winning, below = bears winning.

    ADVANCED USE — RSI Divergence:
      Price makes new HIGH but RSI makes LOWER high → Hidden weakness.
      Price makes new LOW but RSI makes HIGHER low  → Hidden strength.
      These divergences are often the earliest warning signs.

    Indian Market Context:
      In bull markets (like 2021 Nifty rally), RSI stays 60–80 for weeks.
      Don't blindly sell just because RSI > 70 in a strong uptrend.
      Combine with trend direction for context.
    """

    close  = df["Close"].astype(float)
    delta  = close.diff()

    gain   = delta.clip(lower=0)   # only positive changes
    loss   = -delta.clip(upper=0)  # only negative changes (made positive)

    # Wilder's Smoothed Moving Average (not simple average)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi.round(2)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — MACD  (Moving Average Convergence Divergence)
# ══════════════════════════════════════════════════════════════════════

def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Compute MACD Line, Signal Line, and Histogram.

    Formula:
        MACD Line   = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram   = MACD Line - Signal Line

    Args:
        df     : DataFrame with 'Close' column
        fast   : Fast EMA period (default 12)
        slow   : Slow EMA period (default 26)
        signal : Signal line EMA period (default 9)

    Returns:
        pd.DataFrame with columns: MACD, MACD_Signal, MACD_Hist

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why MACD Was Invented
    ──────────────────────────────────────────────
    Gerald Appel created MACD in the 1970s to capture the RELATIONSHIP
    between two different timeframes of momentum.

    EMA(12) = short-term market emotion  (fast-moving, reactive)
    EMA(26) = medium-term market trend   (slower, more stable)

    MACD Line = the GAP between them.
    When the gap WIDENS → momentum is ACCELERATING
    When the gap NARROWS → momentum is FADING

    Reading MACD:
      MACD > 0  → Short-term trend above long-term = BULLISH bias
      MACD < 0  → Short-term trend below long-term = BEARISH bias

    The Signal Line (9-day EMA of MACD) = a "moving average of a moving average"
    It smooths out noise. Crossovers matter:

      MACD crosses ABOVE Signal → BUY signal (momentum turning up)
      MACD crosses BELOW Signal → SELL signal (momentum turning down)

    The Histogram:
      Bars growing taller (positive) → Bulls are gaining momentum
      Bars shrinking (positive→zero) → Momentum is dying = EARLY WARNING
      Bars below zero, growing deeper → Bears are accelerating
      Bars below zero, shrinking back → Bearish momentum fading = Recovery

    HISTOGRAM IS THE EARLIEST SIGNAL.
    Most traders watch price → then MACD line → but smart traders watch
    the HISTOGRAM SLOPE first, before the crossover even happens.
    """

    close       = df["Close"].astype(float)
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()

    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame({
        "MACD":        macd_line.round(4),
        "MACD_Signal": signal_line.round(4),
        "MACD_Hist":   histogram.round(4),
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — SIMPLE MOVING AVERAGES (SMA)
# ══════════════════════════════════════════════════════════════════════

def compute_sma(df: pd.DataFrame, windows: list = [20, 50]) -> pd.DataFrame:
    """
    Compute Simple Moving Averages for given window sizes.

    Args:
        df      : DataFrame with 'Close' column
        windows : List of lookback periods (default [20, 50])

    Returns:
        pd.DataFrame with columns: SMA_20, SMA_50 (or other windows)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Moving Averages Matter
    ──────────────────────────────────────────────
    Daily stock prices are NOISY. One bad day can panic retail investors.
    Moving averages SMOOTH OUT noise to reveal the underlying trend.

    The 20-Day SMA (roughly 1 trading month):
      Tracks SHORT-TERM trend. This is the "tactical" view.
      Active traders use this as dynamic support/resistance.
      Price above SMA_20 → short-term bullish momentum
      Price below SMA_20 → short-term bearish momentum

    The 50-Day SMA (roughly 2.5 trading months):
      Tracks MEDIUM-TERM trend. This is the "strategic" view.
      Institutional investors (mutual funds, FIIs) watch this closely.
      Price above SMA_50 → medium-term uptrend intact
      Price below SMA_50 → medium-term downtrend in force

    THE GOLDEN CROSS (most famous signal in technical analysis):
      SMA_20 crosses ABOVE SMA_50 → Strong bullish signal
      → Short-term momentum aligning with medium-term trend
      → This is what sparked many Nifty rallies historically

    THE DEATH CROSS:
      SMA_20 crosses BELOW SMA_50 → Strong bearish signal
      → Short-term weakness becoming structural downtrend

    KEY INSIGHT — Moving Averages as Dynamic Support:
      In an uptrend, price often dips TO the 20DMA and bounces.
      That dip is a buying opportunity, not a sign of weakness.
      This is why professional traders "buy the dip to the moving average."
    """

    result = {}
    close  = df["Close"].astype(float)

    for window in windows:
        col_name         = f"SMA_{window}"
        result[col_name] = close.rolling(window=window, min_periods=window).mean().round(2)

    return pd.DataFrame(result, index=df.index)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — BOLLINGER BANDS
# ══════════════════════════════════════════════════════════════════════

def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands: Upper, Middle (SMA), Lower.

    Formula:
        Middle Band = SMA(20)
        Upper Band  = SMA(20) + 2 × StdDev(20)
        Lower Band  = SMA(20) - 2 × StdDev(20)
        %B          = (Close - Lower) / (Upper - Lower)  [position within bands]
        Bandwidth   = (Upper - Lower) / Middle            [band width / volatility]

    Args:
        df      : DataFrame with 'Close' column
        period  : Lookback period (default 20)
        num_std : Number of standard deviations (default 2.0)

    Returns:
        pd.DataFrame with: BB_Upper, BB_Middle, BB_Lower, BB_PctB, BB_Width

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Bollinger Bands Were Invented
    ──────────────────────────────────────────────
    John Bollinger created these bands in the 1980s to solve one problem:
    "Is today's price movement NORMAL or ABNORMAL?"

    The bands are built from STANDARD DEVIATION — the statistical measure
    of how spread out prices have been. This makes bands DYNAMIC:
    They WIDEN during volatile periods and NARROW during calm periods.

    Key principles:

    1. BAND TOUCH ≠ SIGNAL
       Price touching upper band alone is NOT a sell signal.
       In strong uptrends, price can "walk the upper band" for weeks.
       (Like Nifty 50 during the 2021 bull run.)

    2. SQUEEZE → EXPLOSION
       When bands narrow tightly (BB_Width very low) = volatility compressed.
       This is a coiled spring. A BIG move is coming — direction TBD.
       The first candle that breaks out of the squeeze = direction signal.
       This is called the "Bollinger Squeeze" — one of the most reliable setups.

    3. %B (Percent B):
       %B = 1.0 → Price at upper band (stretched high)
       %B = 0.5 → Price at middle band (neutral)
       %B = 0.0 → Price at lower band (stretched low)
       %B > 1.0 → Price ABOVE upper band = extreme momentum (breakout territory)
       %B < 0.0 → Price BELOW lower band = panic selling (potential reversal)

    4. MEAN REVERSION vs MOMENTUM:
       Most of the time (70%), price stays within bands → Mean reversion works.
       During trends, price rides the band → Momentum works.
       The ART is knowing which regime you're in. Volume helps decide.
    """

    close       = df["Close"].astype(float)
    middle      = close.rolling(window=period, min_periods=period).mean()
    std_dev     = close.rolling(window=period, min_periods=period).std()

    upper       = middle + (num_std * std_dev)
    lower       = middle - (num_std * std_dev)

    # %B: where is price relative to the bands? 0=lower, 1=upper
    pct_b       = (close - lower) / (upper - lower).replace(0, np.nan)

    # Bandwidth: how wide are the bands relative to middle? measures volatility
    bandwidth   = (upper - lower) / middle.replace(0, np.nan)

    return pd.DataFrame({
        "BB_Upper":  upper.round(2),
        "BB_Middle": middle.round(2),
        "BB_Lower":  lower.round(2),
        "BB_PctB":   pct_b.round(4),
        "BB_Width":  bandwidth.round(4),
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — VOLUME MOVING AVERAGE
# ══════════════════════════════════════════════════════════════════════

def compute_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute volume-based indicators.

    Returns:
        Vol_SMA    : 20-day average volume (the "normal" baseline)
        Vol_Ratio  : Today's volume / average volume
        Vol_Spike  : Boolean — is today's volume 1.5x+ the average?

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Volume is the Truth Serum
    ──────────────────────────────────────────────
    Volume is the ONE indicator that cannot be faked by price manipulation.
    Price can be moved with a small number of trades in illiquid stocks.
    But VOLUME reveals whether the BIG MONEY is actually participating.

    The fundamental rule: PRICE + VOLUME tell a complete story.

    4 combinations to understand deeply:

    1. Price UP  + Volume UP  → CONFIRMED UPTREND ✅
       Buyers are confident and multiplying. Smart money is buying.
       This is the healthiest pattern. Trust the move.

    2. Price UP  + Volume DOWN → WEAK RALLY ⚠️
       Price rising but no one is excited. Retail chasing, institutions absent.
       The rally is likely to fade. "Buying on thin air."

    3. Price DOWN + Volume UP  → CONFIRMED DOWNTREND / CAPITULATION ❌
       Sellers are panicking and multiplying. Institutions may be exiting.
       If this is EXTREME volume, it might be "capitulation" (final sell-off).

    4. Price DOWN + Volume DOWN → WEAK PULLBACK 🟡
       Price dipping but sellers aren't motivated. This is a HEALTHY correction.
       Smart traders use these low-volume dips to BUY in uptrends.

    Vol_Ratio interpretation:
      > 2.0 → Extreme volume spike (news, results, index rebalancing)
      > 1.5 → Volume breakout (significant institutional activity)
      1.0   → Average day
      < 0.7 → Low conviction, avoid reading too much into price move
    """

    volume   = df["Volume"].astype(float)
    vol_sma  = volume.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_sma.replace(0, np.nan)

    return pd.DataFrame({
        "Vol_SMA":   vol_sma.round(0),
        "Vol_Ratio": vol_ratio.round(3),
        "Vol_Spike": (vol_ratio >= 1.5).astype(int),  # 1 = spike, 0 = normal
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — add_all_indicators()
# ══════════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the stock DataFrame in one call.

    This is the main entry point used by the app and rules engine.

    Args:
        df : Raw OHLCV DataFrame from data_fetch.py

    Returns:
        Enriched DataFrame with all indicator columns appended.

    Column reference:
        RSI           → Momentum oscillator (0–100)
        MACD          → MACD line
        MACD_Signal   → Signal line
        MACD_Hist     → Histogram (momentum acceleration)
        SMA_20        → 20-day simple moving average
        SMA_50        → 50-day simple moving average
        BB_Upper      → Bollinger upper band
        BB_Middle     → Bollinger middle band (= SMA_20)
        BB_Lower      → Bollinger lower band
        BB_PctB       → % position within bands (0–1)
        BB_Width      → Band width (volatility proxy)
        Vol_SMA       → 20-day average volume
        Vol_Ratio     → Today's volume / average
        Vol_Spike     → 1 if volume spike, else 0
    """

    enriched = df.copy()

    # ── RSI ──
    enriched["RSI"] = compute_rsi(df)

    # ── MACD ──
    macd_df = compute_macd(df)
    enriched = pd.concat([enriched, macd_df], axis=1)

    # ── Moving Averages ──
    sma_df = compute_sma(df, windows=[20, 50])
    enriched = pd.concat([enriched, sma_df], axis=1)

    # ── Bollinger Bands ──
    bb_df = compute_bollinger_bands(df)
    enriched = pd.concat([enriched, bb_df], axis=1)

    # ── Volume Indicators ──
    vol_df = compute_volume_indicators(df)
    enriched = pd.concat([enriched, vol_df], axis=1)

    return enriched


# ══════════════════════════════════════════════════════════════════════
#  HELPER — get_latest_indicator_snapshot()
# ══════════════════════════════════════════════════════════════════════

def get_latest_indicator_snapshot(df: pd.DataFrame) -> dict:
    """
    Extract the most recent indicator values as a clean dictionary.

    Used by the rules engine and ML model as feature input.
    Also used by the dashboard for the "signal summary" panel.

    Returns dict with the latest value of every indicator.
    """

    row = df.iloc[-1]  # most recent trading day

    def safe(val):
        """Convert to float safely, return None if NaN."""
        try:
            f = float(val)
            return None if np.isnan(f) else round(f, 4)
        except Exception:
            return None

    return {
        # Price
        "close":       safe(row["Close"]),
        "open":        safe(row["Open"]),
        "high":        safe(row["High"]),
        "low":         safe(row["Low"]),

        # Momentum
        "rsi":         safe(row.get("RSI")),

        # Trend
        "macd":        safe(row.get("MACD")),
        "macd_signal": safe(row.get("MACD_Signal")),
        "macd_hist":   safe(row.get("MACD_Hist")),

        # Moving Averages
        "sma_20":      safe(row.get("SMA_20")),
        "sma_50":      safe(row.get("SMA_50")),

        # Bollinger Bands
        "bb_upper":    safe(row.get("BB_Upper")),
        "bb_lower":    safe(row.get("BB_Lower")),
        "bb_pctb":     safe(row.get("BB_PctB")),
        "bb_width":    safe(row.get("BB_Width")),

        # Volume
        "volume":      safe(row["Volume"]),
        "vol_sma":     safe(row.get("Vol_SMA")),
        "vol_ratio":   safe(row.get("Vol_Ratio")),
        "vol_spike":   int(row.get("Vol_Spike", 0)),
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN — self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_fetch import fetch_stock_data

    print("=" * 65)
    print("  STOCK ANALYSIS ENGINE — Indicators Module Test")
    print("=" * 65 + "\n")

    df_raw     = fetch_stock_data("INFY.NS")
    df_enriched = add_all_indicators(df_raw)

    print("── Indicator Columns Added ──")
    indicator_cols = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "SMA_20", "SMA_50", "BB_Upper", "BB_Lower",
        "BB_PctB", "BB_Width", "Vol_Ratio", "Vol_Spike"
    ]
    print(df_enriched[indicator_cols].tail(3).to_string())
    print()

    snap = get_latest_indicator_snapshot(df_enriched)
    print("── Latest Indicator Snapshot ──")
    for k, v in snap.items():
        print(f"   {k:<14}: {v}")

    # ── Human-readable market read ──
    print("\n── 🧠 Indicator-Based Market Read ──")

    rsi = snap["rsi"] or 50
    if rsi > 70:
        print(f"   ⚠️  RSI {rsi:.1f} — Overbought zone. Momentum stretched.")
    elif rsi < 30:
        print(f"   🟢 RSI {rsi:.1f} — Oversold zone. Potential bounce.")
    else:
        print(f"   📊 RSI {rsi:.1f} — Neutral momentum.")

    macd     = snap["macd"]     or 0
    macd_sig = snap["macd_signal"] or 0
    if macd > macd_sig:
        print(f"   📈 MACD above signal — Bullish momentum.")
    else:
        print(f"   📉 MACD below signal — Bearish momentum.")

    close  = snap["close"]  or 0
    sma_20 = snap["sma_20"] or 0
    sma_50 = snap["sma_50"] or 0
    if close > sma_20 > sma_50:
        print(f"   🚀 Price > SMA20 > SMA50 — Classic bullish alignment.")
    elif close < sma_20 < sma_50:
        print(f"   🔻 Price < SMA20 < SMA50 — Classic bearish alignment.")
    else:
        print(f"   〰️  Mixed MA alignment — No clear trend.")

    vol_ratio = snap["vol_ratio"] or 1.0
    if vol_ratio > 1.5:
        print(f"   🔥 Volume {vol_ratio:.1f}x average — Institutional activity detected.")
    else:
        print(f"   📊 Volume {vol_ratio:.1f}x average — Normal trading day.")
