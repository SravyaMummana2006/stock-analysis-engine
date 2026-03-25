"""
╔══════════════════════════════════════════════════════════════════════╗
║              STOCK ANALYSIS ENGINE — data_fetch.py                  ║
║                  Module 1: Data Acquisition Layer                    ║
╚══════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Fetch live and historical OHLCV (Open, High, Low, Close, Volume) data
    for Indian NSE stocks using Yahoo Finance.

STOCK MARKET INSIGHT:
    Every technical indicator — RSI, MACD, Bollinger Bands — is derived
    from price and volume history. Without clean data, analysis is noise.

    Think of this module as the "Bloomberg Terminal" data feed for our engine.
    It answers the first question every analyst asks:
    "What has this stock been doing?"
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# Indian NSE stocks we support (Yahoo Finance uses .NS suffix for NSE)
SUPPORTED_STOCKS = {
    "RELIANCE.NS":    "Reliance Industries",
    "INFY.NS":        "Infosys",
    "SBIN.NS":  "State Bank of India",
}

# How many calendar days of history to fetch
# 365 days gives us enough runway for 200-day moving averages + RSI warmup
DEFAULT_PERIOD_DAYS = 365


# ──────────────────────────────────────────────
# CORE FUNCTION: fetch_stock_data()
# ──────────────────────────────────────────────

def fetch_stock_data(ticker: str, period_days: int = DEFAULT_PERIOD_DAYS) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given NSE stock ticker.

    Args:
        ticker      : Yahoo Finance ticker symbol e.g. "RELIANCE.NS"
        period_days : Number of calendar days of history to fetch

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (trading days only, no weekends/holidays)

    Raises:
        ValueError: if ticker is unsupported or data is empty

    MARKET INSIGHT:
        OHLCV = the 5 DNA strands of every trading day:
          Open   → Where price started (sentiment at market open)
          High   → Maximum optimism of the day (buyers' peak power)
          Low    → Maximum fear of the day (sellers' peak power)
          Close  → Final verdict — the most important price
          Volume → How many shares traded (conviction behind the move)

        A stock moving UP on HIGH volume = strong conviction.
        A stock moving UP on LOW volume  = weak, suspect move.
        This is why volume data is fetched alongside price.
    """

    if ticker not in SUPPORTED_STOCKS:
        raise ValueError(
            f"Ticker '{ticker}' not in supported list. "
            f"Choose from: {list(SUPPORTED_STOCKS.keys())}"
        )

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    print(f"📡 Fetching {SUPPORTED_STOCKS[ticker]} ({ticker}) ...")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

    try:
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,       # suppress yfinance download bar
            auto_adjust=True,     # adjust for splits and dividends automatically
        )
    except Exception as e:
        raise ConnectionError(f"Failed to fetch data for {ticker}: {e}")

    if raw.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            "Check your internet connection or try again."
        )

    # ── Flatten MultiIndex columns if present (yfinance v0.2+ quirk) ──
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ── Keep only OHLCV columns ──
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    df = raw[ohlcv_columns].copy()

    # ── Ensure index is DatetimeIndex ──
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # ── Drop rows where Close is NaN (can happen around splits) ──
    df = df.dropna(subset=["Close"])

    # ── Add human-readable metadata columns ──
    df["Ticker"]   = ticker
    df["Company"]  = SUPPORTED_STOCKS[ticker]

    print(f"   ✅ {len(df)} trading days loaded. "
          f"Latest close: ₹{df['Close'].iloc[-1]:.2f}\n")

    return df


# ──────────────────────────────────────────────
# HELPER: fetch_multiple_stocks()
# ──────────────────────────────────────────────

def fetch_multiple_stocks(
    tickers: list = None,
    period_days: int = DEFAULT_PERIOD_DAYS
) -> dict:
    """
    Fetch data for multiple stocks and return as a dictionary.

    Args:
        tickers     : List of ticker symbols. Defaults to all supported stocks.
        period_days : Days of history to fetch for each stock.

    Returns:
        dict[ticker_symbol -> pd.DataFrame]

    MARKET INSIGHT:
        Comparing multiple stocks simultaneously is called "sector analysis."
        If RELIANCE and TATAMOTORS both drop together, it may be a market-wide
        event (FII selling, rate hike) rather than stock-specific news.
        If only one stock drops while others rise, that's stock-specific risk.
    """

    if tickers is None:
        tickers = list(SUPPORTED_STOCKS.keys())

    stock_data = {}
    failed     = []

    for ticker in tickers:
        try:
            stock_data[ticker] = fetch_stock_data(ticker, period_days)
        except Exception as e:
            print(f"   ⚠️  Skipping {ticker}: {e}")
            failed.append(ticker)

    if failed:
        print(f"\n⚠️  Failed to fetch: {failed}")

    print(f"✅ Successfully loaded {len(stock_data)} stock(s).\n")
    return stock_data


# ──────────────────────────────────────────────
# HELPER: get_stock_summary()
# ──────────────────────────────────────────────

def get_stock_summary(df: pd.DataFrame) -> dict:
    """
    Generate a quick statistical summary of the stock data.

    Returns a dictionary with key price and volume statistics.

    MARKET INSIGHT:
        Before any technical analysis, an analyst does a "sanity check":
          - What's the 52-week range? (How volatile is this stock?)
          - What's the average volume? (How liquid is it?)
          - How far is current price from 52-week high? (Is it beaten down?)

        A stock near its 52-week LOW with rising volume = accumulation signal.
        A stock near its 52-week HIGH with rising volume = breakout candidate.
    """

    close   = df["Close"]
    volume  = df["Volume"]

    latest_close    = close.iloc[-1]
    week52_high     = close.max()
    week52_low      = close.min()
    avg_volume      = volume.mean()
    latest_volume   = volume.iloc[-1]
    pct_from_high   = ((latest_close - week52_high) / week52_high) * 100
    pct_from_low    = ((latest_close - week52_low)  / week52_low)  * 100
    volume_ratio    = latest_volume / avg_volume  # > 1.5 = volume breakout

    return {
        "ticker":          df["Ticker"].iloc[-1],
        "company":         df["Company"].iloc[-1],
        "latest_close":    round(float(latest_close), 2),
        "week52_high":     round(float(week52_high), 2),
        "week52_low":      round(float(week52_low), 2),
        "pct_from_high":   round(float(pct_from_high), 2),
        "pct_from_low":    round(float(pct_from_low), 2),
        "avg_daily_volume": int(avg_volume),
        "latest_volume":   int(latest_volume),
        "volume_ratio":    round(float(volume_ratio), 2),
        "data_start":      df.index[0].strftime("%Y-%m-%d"),
        "data_end":        df.index[-1].strftime("%Y-%m-%d"),
        "total_days":      len(df),
    }


# ──────────────────────────────────────────────
# MAIN — quick test / demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  STOCK ANALYSIS ENGINE — Data Fetch Module Test")
    print("=" * 60 + "\n")

    # Test single stock
    df = fetch_stock_data("RELIANCE.NS")

    print("── Sample Data (last 5 rows) ──")
    print(df[["Open", "High", "Low", "Close", "Volume"]].tail())
    print()

    # Summary stats
    summary = get_stock_summary(df)
    print("── Stock Summary ──")
    for key, value in summary.items():
        print(f"   {key:<22}: {value}")
    print()

    # ── Key market insight from this data ──
    vr = summary["volume_ratio"]
    pct_high = summary["pct_from_high"]

    print("── 📊 Quick Market Read ──")
    if vr > 1.5:
        print(f"   🔥 Volume breakout! Today's volume is {vr:.1f}x the average.")
    elif vr < 0.7:
        print(f"   😴 Low volume day ({vr:.1f}x avg). Weak conviction in price move.")
    else:
        print(f"   📊 Normal volume day ({vr:.1f}x avg).")

    if pct_high > -5:
        print(f"   🚀 Price is near 52-week high ({pct_high:.1f}% away). Momentum territory.")
    elif pct_high < -20:
        print(f"   📉 Price is {abs(pct_high):.1f}% below 52-week high. Possible value zone.")
    else:
        print(f"   📈 Price is {abs(pct_high):.1f}% below 52-week high. Mid-range.")
