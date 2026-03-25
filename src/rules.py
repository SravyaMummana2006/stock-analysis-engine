"""
╔══════════════════════════════════════════════════════════════════════╗
║               STOCK ANALYSIS ENGINE — rules.py                      ║
║         Module 3: Market Intelligence & Signal Detection             ║
╚══════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Transform raw indicator values into intelligent market signals.
    This module is the "analyst brain" — it looks at all indicators
    together and produces structured conclusions:

      1. Trend Classification   → Uptrend / Downtrend / Sideways
      2. Support & Resistance   → Dynamic price floor and ceiling levels
      3. Volume Breakout        → Real institutional move vs. thin-air rally
      4. Failed Breakout        → Trap detection for retail investors
      5. Momentum Strength      → Scoring how powerful the current move is
      6. Analyst Explanation    → Human-readable reasoning for every signal

PHILOSOPHY:
    Rules are not rigid — they are weighted observations.
    A stock with 6/7 bullish signals is more trustworthy than one with 3/7.
    This module produces a "conviction score" alongside every signal.

    Think of it as a junior analyst's checklist that an experienced
    portfolio manager reviews before making a trade decision.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE — StockSignal
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StockSignal:
    """
    Container for all analysis outputs for a single stock.

    Every field represents one layer of the analyst's view.
    The dashboard reads this object to render the complete signal panel.
    """

    ticker:              str   = ""
    company:             str   = ""

    # ── Trend ──
    trend:               str   = "Unknown"       # "Uptrend" | "Downtrend" | "Sideways"
    trend_strength:      str   = "Weak"          # "Strong" | "Moderate" | "Weak"

    # ── Support / Resistance ──
    support:             float = 0.0             # key price floor
    resistance:          float = 0.0             # key price ceiling
    support_method:      str   = ""              # how it was calculated
    resistance_method:   str   = ""

    # ── Breakout Analysis ──
    volume_breakout:     bool  = False           # True = volume spike with price move
    breakout_direction:  str   = "None"          # "Bullish" | "Bearish" | "None"
    failed_breakout:     bool  = False           # True = price broke out but reversed
    failed_breakout_dir: str   = "None"          # direction of the failed breakout

    # ── Momentum ──
    momentum_score:      int   = 0               # -5 to +5 (negative=bear, positive=bull)
    momentum_label:      str   = "Neutral"       # "Strong Bull" | "Bull" | "Neutral" | ...

    # ── Individual Signal Flags ──
    price_above_sma20:   bool  = False
    price_above_sma50:   bool  = False
    golden_cross:        bool  = False           # SMA20 > SMA50 (recent)
    death_cross:         bool  = False           # SMA20 < SMA50 (recent)
    rsi_overbought:      bool  = False           # RSI > 70
    rsi_oversold:        bool  = False           # RSI < 30
    rsi_bullish_zone:    bool  = False           # RSI 50-70 (healthy bull)
    macd_bullish:        bool  = False           # MACD > Signal
    macd_hist_rising:    bool  = False           # Histogram increasing
    bb_squeeze:          bool  = False           # Volatility compression
    bb_upper_touch:      bool  = False           # Price near upper band
    bb_lower_touch:      bool  = False           # Price near lower band
    vol_above_avg:       bool  = False           # Volume > average

    # ── Analyst Explanation ──
    explanation:         str   = ""              # full analyst-style reasoning
    signal_tags:         list  = field(default_factory=list)   # short signal chips


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — TREND DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_trend(df: pd.DataFrame, snap: dict) -> tuple:
    """
    Classify the stock's trend using a multi-layer approach.

    Layer 1: Price vs Moving Averages (structural trend)
    Layer 2: Moving Average slope (is the trend accelerating?)
    Layer 3: Recent price action (last 20 days higher highs / lower lows)

    Returns:
        (trend: str, strength: str)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: The 3-Layer Trend Framework
    ──────────────────────────────────────────────
    Most retail traders look at a chart and say "it's going up."
    Professionals use a STRUCTURED framework to classify trend quality.

    LAYER 1 — Moving Average Stack:
        Bullish stack: Price > SMA20 > SMA50
          → Short-term AND medium-term trend aligned upward
          → This is the "textbook uptrend" condition
        Bearish stack: Price < SMA20 < SMA50
          → Both timeframes trending down — high-conviction downtrend
        Mixed: Any other combination → Sideways / transitioning

    LAYER 2 — MA Slope:
        Is SMA20 rising or falling over the last 5 days?
        A rising SMA20 means the uptrend is actively accelerating.
        A flattening SMA20 means the trend is losing energy.

    LAYER 3 — Higher Highs / Lower Lows:
        In a genuine uptrend: each rally peak is higher than the last.
        In a genuine downtrend: each pullback low is lower than the last.
        This is the oldest definition of trend — used since Charles Dow (1900s).

    TREND STRENGTH:
        Strong   = All 3 layers agree
        Moderate = 2 of 3 layers agree
        Weak     = Only 1 layer, or conflicting signals
    """

    close  = snap["close"]  or 0
    sma20  = snap["sma_20"] or 0
    sma50  = snap["sma_50"] or 0

    score = 0  # positive = bullish, negative = bearish

    # ── Layer 1: MA Stack ──
    price_above_20  = close > sma20
    price_above_50  = close > sma50
    ma_stack_bull   = sma20 > sma50   # golden stack

    if price_above_20:  score += 1
    if price_above_50:  score += 1
    if ma_stack_bull:   score += 1

    # ── Layer 2: SMA20 Slope (last 5 days) ──
    if "SMA_20" in df.columns:
        sma20_series = df["SMA_20"].dropna()
        if len(sma20_series) >= 6:
            sma20_slope = sma20_series.iloc[-1] - sma20_series.iloc[-6]
            if sma20_slope > 0:    score += 1
            elif sma20_slope < 0:  score -= 1

    # ── Layer 3: Higher Highs / Lower Lows (last 20 days) ──
    if len(df) >= 20:
        recent      = df["Close"].iloc[-20:].values
        first_half  = recent[:10]
        second_half = recent[10:]
        hh = second_half.max() > first_half.max()   # higher high
        ll = second_half.min() < first_half.min()   # lower low

        if hh and not ll:  score += 1   # higher highs, no lower lows = uptrend
        if ll and not hh:  score -= 1   # lower lows, no higher highs = downtrend

    # ── Classify Trend ──
    if score >= 3:
        trend, strength = "Uptrend", "Strong"
    elif score == 2:
        trend, strength = "Uptrend", "Moderate"
    elif score == 1:
        trend, strength = "Sideways", "Weak"
    elif score == 0:
        trend, strength = "Sideways", "Moderate"
    elif score == -1:
        trend, strength = "Sideways", "Weak"
    elif score == -2:
        trend, strength = "Downtrend", "Moderate"
    else:
        trend, strength = "Downtrend", "Strong"

    return trend, strength


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — SUPPORT & RESISTANCE DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_support_resistance(df: pd.DataFrame) -> tuple:
    """
    Identify the most relevant support and resistance levels.

    Method:
        Uses a combination of:
        1. Recent swing lows  (local minima)  → support
        2. Recent swing highs (local maxima)  → resistance
        3. SMA levels as dynamic S/R fallback

    Returns:
        (support: float, resistance: float,
         support_method: str, resistance_method: str)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: What Are Support & Resistance?
    ──────────────────────────────────────────────
    Support and Resistance are the TWO most important concepts in all
    of technical analysis. Every professional analyst uses them.

    SUPPORT = A price level where BUYERS historically step in.
    → Below this price, demand exceeds supply → price bounces up.
    → Think of it as a "floor" that the market has tested and held.

    RESISTANCE = A price level where SELLERS historically step in.
    → Above this price, supply exceeds demand → price gets rejected.
    → Think of it as a "ceiling" the market has tried to break through.

    WHY DO THESE LEVELS FORM?
    Human psychology. At ₹2500, thousands of investors bought Reliance.
    If price falls back to ₹2500, those same investors will buy again
    to "add to their position." This collective memory creates support.

    Similarly, at ₹3000, many investors who bought at lower prices
    decide "I'll sell if it gets back to ₹3000." This creates resistance.

    BREAKOUT vs REJECTION:
    When price approaches resistance:
      → Rejected (closes below) = resistance held, possible short trade
      → Breaks above + volume   = resistance broken, becomes NEW support

    THE ROLE REVERSAL PRINCIPLE:
    Once broken, resistance BECOMES support and vice versa.
    This is one of the most reliable patterns in all technical analysis.

    HOW WE DETECT SWING HIGHS/LOWS:
    A swing low  = a candle whose low is lower than 5 candles on each side
    A swing high = a candle whose high is higher than 5 candles on each side
    These are points where price "bounced" — revealing buyer/seller zones.
    """

    highs  = df["High"].values
    lows   = df["Low"].values
    close  = df["Close"].values
    n      = len(df)
    window = 5  # look 5 bars left and right to confirm swing point

    swing_highs = []
    swing_lows  = []

    for i in range(window, n - window):
        local_high = highs[i]
        local_low  = lows[i]

        if local_high == max(highs[i - window: i + window + 1]):
            swing_highs.append(local_high)
        if local_low == min(lows[i - window: i + window + 1]):
            swing_lows.append(local_low)

    current_close = close[-1]
    sma20 = df["SMA_20"].dropna().iloc[-1] if "SMA_20" in df.columns else current_close
    sma50 = df["SMA_50"].dropna().iloc[-1] if "SMA_50" in df.columns else current_close

    # ── Find nearest support (swing low BELOW current price) ──
    support_candidates = [s for s in swing_lows if s < current_close]
    if support_candidates:
        support         = max(support_candidates)  # closest floor below
        support_method  = "Swing Low"
    else:
        support         = min(sma20, sma50)
        support_method  = "Moving Average"

    # ── Find nearest resistance (swing high ABOVE current price) ──
    resistance_candidates = [r for r in swing_highs if r > current_close]
    if resistance_candidates:
        resistance        = min(resistance_candidates)  # closest ceiling above
        resistance_method = "Swing High"
    else:
        resistance        = df["Close"].rolling(20).max().iloc[-1]
        resistance_method = "20-Day High"

    return (
        round(float(support), 2),
        round(float(resistance), 2),
        support_method,
        resistance_method,
    )


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — VOLUME BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_volume_breakout(df: pd.DataFrame, snap: dict) -> tuple:
    """
    Detect if today's price move is confirmed by a volume breakout.

    Returns:
        (is_breakout: bool, direction: str)
        direction: "Bullish" | "Bearish" | "None"

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: The Volume Breakout — What It Really Means
    ──────────────────────────────────────────────
    A volume breakout is when price moves SIGNIFICANTLY above or below
    a key level AND volume is 1.5x or more above the average.

    Why does this matter?
    Price can move on low volume due to a single large order or low liquidity.
    But VOLUME breakouts require MANY participants to agree simultaneously.
    This is the footprint of institutional money — FIIs, mutual funds, HNIs.

    BULLISH VOLUME BREAKOUT:
      → Price breaks above resistance on high volume
      → Signals: new buyers overwhelming sellers at that level
      → Institutions are "accumulating" — building positions
      → Often seen before strong multi-week rallies

    BEARISH VOLUME BREAKOUT:
      → Price breaks below support on high volume
      → Signals: sellers overwhelming buyers
      → Institutions are "distributing" — exiting positions
      → Often precedes sharp sustained declines

    WHY VOLUME BREAKOUTS SUCCEED:
    When price breaks a well-known level with high volume, it creates
    a "self-fulfilling prophecy" — all the traders who were watching
    that level now pile in, amplifying the move.

    WATCH OUT — News Events:
    Volume spikes on result days, RBI policy days, Budget day, etc.
    are not pure technical breakouts. Always cross-reference with news.
    Our engine flags the breakout; context requires human judgment.
    """

    vol_ratio  = snap.get("vol_ratio") or 1.0
    vol_spike  = bool(snap.get("vol_spike", 0))
    close      = snap["close"]  or 0
    sma20      = snap["sma_20"] or 0
    sma50      = snap["sma_50"] or 0

    # Price breaking above a key MA with volume = bullish breakout
    is_breakout   = vol_spike and vol_ratio >= 1.5
    direction     = "None"

    if is_breakout:
        if close > sma20 and close > sma50:
            direction = "Bullish"
        elif close < sma20 and close < sma50:
            direction = "Bearish"
        else:
            direction = "Neutral"  # volume spike but price in MA zone

    return is_breakout, direction


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — FAILED BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_failed_breakout(df: pd.DataFrame, resistance: float, support: float) -> tuple:
    """
    Detect if a recent breakout attempt failed (price reversed).

    Logic:
      Bullish failed breakout: Price broke above resistance in last 3 days
                               but has since closed back below it.
      Bearish failed breakout: Price broke below support in last 3 days
                               but has since closed back above it.

    Returns:
        (failed: bool, direction: str)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: The Failed Breakout — Retail's Biggest Trap
    ──────────────────────────────────────────────
    The failed breakout (also called a "fakeout" or "bull trap") is one of
    the most common and costly mistakes retail traders make.

    THE TRAP SCENARIO:
    1. Stock has been trading below ₹3000 resistance for weeks
    2. One day it BREAKS ABOVE ₹3000 with some excitement
    3. Retail traders FOMO in — "breakout confirmed!"
    4. Price holds for 1–2 days, then reverses back below ₹3000
    5. All the buyers who chased the breakout are now trapped at the top

    WHY DO FAILED BREAKOUTS HAPPEN?
    Theory 1 — Stop Hunt: Large institutions deliberately push price above
    resistance to trigger retail buy orders and short-stop-losses.
    They sell into this artificial demand and price falls back.

    Theory 2 — Lack of Follow-Through: Real breakouts need volume.
    If a breakout happens on low volume, there aren't enough buyers to
    sustain the move, and sellers take the stock back down.

    Theory 3 — Overhead Supply: Many investors bought at resistance during
    the last visit. They've been "waiting to break even" and sell immediately
    when price returns. This creates a "supply overhang."

    HOW TO AVOID THE TRAP:
    → Wait for the CLOSING price above resistance, not intraday
    → Require VOLUME confirmation (our engine enforces this)
    → Wait 2–3 days to see if the level holds as new support
    → The safest entry is after a retest of broken resistance

    Our engine flags this as a WARNING signal — not a trade recommendation.
    It teaches you to ask: "Did this breakout actually follow through?"
    """

    if len(df) < 5:
        return False, "None"

    recent_closes  = df["Close"].iloc[-5:].values
    recent_highs   = df["High"].iloc[-5:].values
    direction      = "None"
    failed         = False

    current_close = recent_closes[-1]
    buffer = 0.005  # 0.5% buffer to account for noise around exact levels

    # ── Check for Bullish Failed Breakout ──
    # Any of last 3 candles closed above resistance but today is back below
    for i in range(-4, -1):
        if recent_closes[i] > resistance * (1 + buffer):
            if current_close < resistance:
                failed    = True
                direction = "Bullish"  # failed bull breakout = bearish signal
                break

    # ── Check for Bearish Failed Breakout ──
    if not failed:
        for i in range(-4, -1):
            if recent_closes[i] < support * (1 - buffer):
                if current_close > support:
                    failed    = True
                    direction = "Bearish"  # failed bear breakdown = bullish signal
                    break

    return failed, direction


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — MOMENTUM SCORING
# ══════════════════════════════════════════════════════════════════════

def score_momentum(snap: dict) -> tuple:
    """
    Generate a momentum score from -5 to +5 by tallying individual signals.

    Each bullish signal adds +1. Each bearish signal adds -1.
    The final score represents "how many indicators agree on direction."

    Returns:
        (score: int, label: str)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Momentum Scoring Works
    ──────────────────────────────────────────────
    No single indicator is right 100% of the time.
    But when MULTIPLE independent indicators all point the same direction,
    the probability of being correct increases significantly.

    This is called "confluence" — the overlap of multiple signals.

    Example of STRONG BULL momentum (score = +5):
      ✅ Price above SMA20          → Short-term bullish structure
      ✅ Price above SMA50          → Medium-term bullish structure
      ✅ RSI between 50–70          → Healthy momentum, not exhausted
      ✅ MACD above Signal          → Momentum building
      ✅ MACD Histogram rising      → Acceleration in progress
      ✅ Volume above average       → Conviction behind the move
    → Score: +5 → "Strong Bull" → High-conviction setup

    Example of CONFLICTED signal (score = 0):
      ✅ Price above SMA20
      ❌ Price below SMA50
      ⚪ RSI at 50 (neutral)
      ❌ MACD below signal
      ✅ Volume above average
    → Score: +1 → "Neutral" → No clear edge either way

    SCORE INTERPRETATION:
      +4 to +5 → Strong Bull  (multiple signals confirming upside)
      +2 to +3 → Bull         (majority bullish, some caution needed)
       0 to +1 → Neutral      (mixed signals, no directional edge)
      -2 to -1 → Bear         (majority bearish signals)
      -4 to -5 → Strong Bear  (high-conviction downside pressure)
    """

    score = 0
    flags = {}

    rsi         = snap.get("rsi")        or 50
    macd        = snap.get("macd")       or 0
    macd_sig    = snap.get("macd_signal") or 0
    macd_hist   = snap.get("macd_hist")  or 0
    close       = snap.get("close")      or 0
    sma20       = snap.get("sma_20")     or 0
    sma50       = snap.get("sma_50")     or 0
    bb_pctb     = snap.get("bb_pctb")    or 0.5
    vol_ratio   = snap.get("vol_ratio")  or 1.0

    # ── Signal 1: Price vs SMA20 ──
    flags["price_above_sma20"] = close > sma20
    score += 1 if flags["price_above_sma20"] else -1

    # ── Signal 2: Price vs SMA50 ──
    flags["price_above_sma50"] = close > sma50
    score += 1 if flags["price_above_sma50"] else -1

    # ── Signal 3: RSI Zone ──
    flags["rsi_overbought"]  = rsi > 70
    flags["rsi_oversold"]    = rsi < 30
    flags["rsi_bullish_zone"] = 50 <= rsi <= 70

    if flags["rsi_bullish_zone"]:
        score += 1    # healthy bullish momentum
    elif rsi > 70:
        score += 0    # overbought: good momentum but exhaustion risk, neutral
    elif rsi < 30:
        score -= 1    # oversold: selling pressure dominant
    else:
        score -= 0    # neutral zone, no contribution

    # ── Signal 4: MACD crossover ──
    flags["macd_bullish"] = macd > macd_sig
    score += 1 if flags["macd_bullish"] else -1

    # ── Signal 5: MACD Histogram direction ──
    # We infer if histogram is rising by checking its sign and magnitude
    flags["macd_hist_rising"] = macd_hist > 0
    score += 1 if flags["macd_hist_rising"] else -1

    # ── Signal 6: Volume conviction ──
    flags["vol_above_avg"] = vol_ratio >= 1.0
    score += 1 if flags["vol_above_avg"] else 0  # low vol doesn't penalise hard

    # ── Signal 7: Bollinger %B position ──
    flags["bb_upper_touch"] = bb_pctb > 0.85
    flags["bb_lower_touch"] = bb_pctb < 0.15

    # Being near upper band in uptrend = strength; in isolation = stretched
    # We weight this lightly — just note it
    if bb_pctb > 0.6:
        score += 0   # leaning bullish but not a clean signal without trend context
    elif bb_pctb < 0.2:
        score -= 1   # near lower band = price under pressure

    # ── Label the score ──
    if score >= 4:    label = "Strong Bull"
    elif score >= 2:  label = "Bull"
    elif score >= 0:  label = "Neutral"
    elif score >= -2: label = "Bear"
    else:             label = "Strong Bear"

    return score, label, flags


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — ANALYST EXPLANATION GENERATOR
# ══════════════════════════════════════════════════════════════════════

def generate_explanation(signal: "StockSignal", snap: dict) -> tuple:
    """
    Generate an analyst-style explanation and signal tag list.

    Returns:
        (explanation: str, tags: list[str])

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Explainability Matters
    ──────────────────────────────────────────────
    A black-box "BUY" signal is dangerous. An analyst doesn't just say
    "this stock will go up" — they explain WHY, so the listener can
    EVALUATE the quality of the reasoning.

    The explanation answers three critical investor questions:
      1. What is the STRUCTURAL condition? (Trend, MAs)
      2. What is the MOMENTUM condition? (RSI, MACD)
      3. What is the VOLUME condition? (Is money flowing in or out?)

    A complete explanation sounds like a morning briefing:
    "RELIANCE is in a confirmed uptrend — price is above both the 20 and 50-day
    moving averages. RSI at 62 indicates healthy momentum, not yet overbought.
    MACD has crossed above its signal line, suggesting momentum is building.
    Volume is 1.8x the average, indicating institutional participation.
    Watch ₹2900 as key support and ₹3100 as resistance."

    This is EXACTLY what our engine generates — structured market logic
    converted into readable English that teaches you how professionals think.
    """

    rsi       = snap.get("rsi")        or 50
    macd      = snap.get("macd")       or 0
    macd_sig  = snap.get("macd_signal") or 0
    close     = snap.get("close")      or 0
    sma20     = snap.get("sma_20")     or 0
    sma50     = snap.get("sma_50")     or 0
    vol_ratio = snap.get("vol_ratio")  or 1.0
    bb_pctb   = snap.get("bb_pctb")   or 0.5
    bb_width  = snap.get("bb_width")  or 0

    parts = []  # individual sentences
    tags  = []  # short chip tags for dashboard

    # ── Trend ──
    if signal.trend == "Uptrend":
        if signal.trend_strength == "Strong":
            parts.append(
                f"Price is in a strong uptrend — trading above both the "
                f"20-day (₹{sma20:.0f}) and 50-day (₹{sma50:.0f}) moving averages "
                f"with a rising MA stack."
            )
            tags.append("🚀 Strong Uptrend")
        else:
            parts.append(
                f"Price shows an uptrend bias — above SMA20 (₹{sma20:.0f}), "
                f"though conviction is moderate."
            )
            tags.append("📈 Uptrend")
    elif signal.trend == "Downtrend":
        if signal.trend_strength == "Strong":
            parts.append(
                f"Price is in a confirmed downtrend — trading below both the "
                f"20-day (₹{sma20:.0f}) and 50-day (₹{sma50:.0f}) moving averages. "
                f"Bearish structure intact."
            )
            tags.append("🔻 Strong Downtrend")
        else:
            parts.append(
                f"Price shows a downtrend bias — below SMA20 (₹{sma20:.0f}). "
                f"Selling pressure present but not extreme."
            )
            tags.append("📉 Downtrend")
    else:
        parts.append(
            f"Price is in a sideways / consolidation phase between "
            f"SMA20 (₹{sma20:.0f}) and SMA50 (₹{sma50:.0f}). "
            f"No clear directional bias yet."
        )
        tags.append("〰️ Sideways")

    # ── RSI ──
    if rsi > 70:
        parts.append(
            f"RSI at {rsi:.1f} is in overbought territory — momentum has been "
            f"strong but may be due for a short-term pullback or consolidation."
        )
        tags.append(f"⚠️ RSI Overbought ({rsi:.0f})")
    elif rsi < 30:
        parts.append(
            f"RSI at {rsi:.1f} is in oversold territory — selling may be overdone. "
            f"Watch for a potential reversal or bounce near support levels."
        )
        tags.append(f"🟢 RSI Oversold ({rsi:.0f})")
    elif 50 <= rsi <= 70:
        parts.append(
            f"RSI at {rsi:.1f} is in the bullish momentum zone (50–70) — "
            f"healthy trend strength without exhaustion risk."
        )
        tags.append(f"✅ RSI Healthy ({rsi:.0f})")
    else:
        parts.append(
            f"RSI at {rsi:.1f} is in neutral territory — no strong momentum signal "
            f"in either direction."
        )
        tags.append(f"📊 RSI Neutral ({rsi:.0f})")

    # ── MACD ──
    if macd > macd_sig:
        hist_size = abs(macd - macd_sig)
        parts.append(
            f"MACD is above its signal line (gap: {hist_size:.3f}) — "
            f"bullish momentum is building. Trend direction confirmed by MACD."
        )
        tags.append("📈 MACD Bullish")
    else:
        hist_size = abs(macd - macd_sig)
        parts.append(
            f"MACD is below its signal line (gap: {hist_size:.3f}) — "
            f"bearish momentum is dominant. Exercise caution on long trades."
        )
        tags.append("📉 MACD Bearish")

    # ── Volume ──
    if vol_ratio >= 2.0:
        parts.append(
            f"Volume is {vol_ratio:.1f}x above average — an exceptional spike "
            f"suggesting strong institutional activity. Treat this as a high-conviction "
            f"signal in the direction of the price move."
        )
        tags.append(f"🔥 Volume Spike ({vol_ratio:.1f}x)")
    elif vol_ratio >= 1.5:
        parts.append(
            f"Volume is {vol_ratio:.1f}x above average — above-average participation "
            f"confirming the current price move."
        )
        tags.append(f"📊 Volume Above Avg ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.7:
        parts.append(
            f"Volume is only {vol_ratio:.1f}x the average — low conviction. "
            f"Price movement today lacks institutional participation."
        )
        tags.append(f"😴 Low Volume ({vol_ratio:.1f}x)")
    else:
        parts.append(
            f"Volume is {vol_ratio:.1f}x the average — normal trading activity."
        )

    # ── Support & Resistance ──
    parts.append(
        f"Key support at ₹{signal.support:.0f} ({signal.support_method}). "
        f"Key resistance at ₹{signal.resistance:.0f} ({signal.resistance_method})."
    )

    # ── Volume Breakout ──
    if signal.volume_breakout:
        if signal.breakout_direction == "Bullish":
            parts.append(
                f"⚡ A bullish volume breakout has been detected — price is above "
                f"key moving averages with strong volume confirmation. "
                f"This suggests institutional accumulation."
            )
            tags.append("⚡ Volume Breakout (Bull)")
        elif signal.breakout_direction == "Bearish":
            parts.append(
                f"⚡ A bearish volume breakdown has been detected — price is below "
                f"moving averages with high volume. Suggests distribution / selling pressure."
            )
            tags.append("⚡ Volume Breakdown (Bear)")

    # ── Failed Breakout ──
    if signal.failed_breakout:
        if signal.failed_breakout_dir == "Bullish":
            parts.append(
                f"⚠️ A failed bullish breakout is detected — price briefly broke "
                f"above resistance (₹{signal.resistance:.0f}) but has closed back "
                f"below it. This 'bull trap' pattern is bearish. Exercise caution."
            )
            tags.append("⚠️ Failed Breakout")
        elif signal.failed_breakout_dir == "Bearish":
            parts.append(
                f"🟢 A failed bearish breakdown is detected — price dipped below "
                f"support (₹{signal.support:.0f}) but recovered above it. "
                f"This 'bear trap' reversal is bullish."
            )
            tags.append("🟢 Failed Breakdown (Bull Signal)")

    # ── Bollinger Squeeze ──
    if signal.bb_squeeze:
        parts.append(
            f"⚡ Bollinger Band squeeze detected (low volatility compression). "
            f"A significant price move is building — watch for a directional breakout."
        )
        tags.append("⚡ BB Squeeze")

    explanation = " ".join(parts)
    return explanation, tags


# ══════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — analyze_stock()
# ══════════════════════════════════════════════════════════════════════

def analyze_stock(df: pd.DataFrame, snap: dict, ticker: str = "", company: str = "") -> StockSignal:
    """
    Run the complete rules engine on an enriched DataFrame.

    This is the single entry point used by the app and ML model.

    Args:
        df      : Enriched DataFrame (with all indicators from indicators.py)
        snap    : Latest indicator snapshot dict
        ticker  : Stock ticker symbol
        company : Company name

    Returns:
        StockSignal dataclass with all analysis results populated
    """

    sig = StockSignal(ticker=ticker, company=company)

    # ── 1. Trend ──
    sig.trend, sig.trend_strength = detect_trend(df, snap)

    # ── 2. Support & Resistance ──
    sig.support, sig.resistance, sig.support_method, sig.resistance_method = \
        detect_support_resistance(df)

    # ── 3. Volume Breakout ──
    sig.volume_breakout, sig.breakout_direction = detect_volume_breakout(df, snap)

    # ── 4. Failed Breakout ──
    sig.failed_breakout, sig.failed_breakout_dir = \
        detect_failed_breakout(df, sig.resistance, sig.support)

    # ── 5. Momentum Score ──
    sig.momentum_score, sig.momentum_label, flags = score_momentum(snap)

    # ── Propagate flags to signal object ──
    sig.price_above_sma20  = flags.get("price_above_sma20",  False)
    sig.price_above_sma50  = flags.get("price_above_sma50",  False)
    sig.rsi_overbought     = flags.get("rsi_overbought",     False)
    sig.rsi_oversold       = flags.get("rsi_oversold",       False)
    sig.rsi_bullish_zone   = flags.get("rsi_bullish_zone",   False)
    sig.macd_bullish       = flags.get("macd_bullish",       False)
    sig.macd_hist_rising   = flags.get("macd_hist_rising",   False)
    sig.vol_above_avg      = flags.get("vol_above_avg",      False)
    sig.bb_upper_touch     = flags.get("bb_upper_touch",     False)
    sig.bb_lower_touch     = flags.get("bb_lower_touch",     False)

    # ── Bollinger Squeeze ──
    bb_width = snap.get("bb_width") or 0.1
    sig.bb_squeeze = bb_width < 0.04  # very narrow bands = squeeze

    # ── Golden / Death Cross ──
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        sma20_series = df["SMA_20"].dropna()
        sma50_series = df["SMA_50"].dropna()
        if len(sma20_series) >= 5 and len(sma50_series) >= 5:
            recent_20 = sma20_series.iloc[-5:]
            recent_50 = sma50_series.iloc[-5:]
            crosses_above = (recent_20.iloc[-1] > recent_50.iloc[-1]) and \
                            (recent_20.iloc[0]  < recent_50.iloc[0])
            crosses_below = (recent_20.iloc[-1] < recent_50.iloc[-1]) and \
                            (recent_20.iloc[0]  > recent_50.iloc[0])
            sig.golden_cross = crosses_above
            sig.death_cross  = crosses_below

    # ── 6. Explanation + Tags ──
    sig.explanation, sig.signal_tags = generate_explanation(sig, snap)

    return sig


# ══════════════════════════════════════════════════════════════════════
#  MAIN — self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_fetch   import fetch_stock_data
    from indicators   import add_all_indicators, get_latest_indicator_snapshot

    print("=" * 65)
    print("  STOCK ANALYSIS ENGINE — Rules Engine Test")
    print("=" * 65 + "\n")

    for ticker in ["RELIANCE.NS", "INFY.NS", "SBIN.NS"]:
        df_raw      = fetch_stock_data(ticker)
        df_enriched = add_all_indicators(df_raw)
        snap        = get_latest_indicator_snapshot(df_enriched)
        company     = df_raw["Company"].iloc[-1]

        signal = analyze_stock(df_enriched, snap, ticker=ticker, company=company)

        print(f"{'─'*60}")
        print(f"  {signal.company} ({signal.ticker})")
        print(f"{'─'*60}")
        print(f"  Trend         : {signal.trend} ({signal.trend_strength})")
        print(f"  Momentum      : {signal.momentum_label} (score: {signal.momentum_score:+d})")
        print(f"  Support       : ₹{signal.support:,.2f}  [{signal.support_method}]")
        print(f"  Resistance    : ₹{signal.resistance:,.2f}  [{signal.resistance_method}]")
        print(f"  Vol Breakout  : {'✅ Yes — ' + signal.breakout_direction if signal.volume_breakout else '❌ No'}")
        print(f"  Failed BO     : {'⚠️  Yes — ' + signal.failed_breakout_dir if signal.failed_breakout else '✅ No'}")
        print(f"  BB Squeeze    : {'⚡ Yes' if signal.bb_squeeze else 'No'}")
        print(f"  Golden Cross  : {'✅' if signal.golden_cross else '—'}")
        print(f"  Death Cross   : {'⚠️' if signal.death_cross else '—'}")
        print()
        print(f"  Signal Tags   : {' | '.join(signal.signal_tags)}")
        print()
        print(f"  📋 Analyst View:")
        # Wrap explanation for readable console output
        words = signal.explanation.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 72:
                lines.append("     " + " ".join(line[:-1]))
                line = [w]
        if line:
            lines.append("     " + " ".join(line))
        print("\n".join(lines))
        print()
