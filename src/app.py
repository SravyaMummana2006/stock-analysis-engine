"""
╔══════════════════════════════════════════════════════════════════════╗
║                STOCK ANALYSIS ENGINE — app.py                       ║
║              Module 5: Professional Streamlit Dashboard              ║
╚══════════════════════════════════════════════════════════════════════╝

PURPOSE:
    The final integration layer. Ties together all 4 modules into a
    single, interactive, investor-grade dashboard that displays:

      1. Candlestick chart with indicators overlaid
      2. RSI, MACD, Volume sub-charts
      3. Signal summary panel (trend, momentum, breakout)
      4. ML classification result with confidence bars
      5. Feature importance chart
      6. Full analyst explanation text
      7. Support & resistance levels
      8. Raw data table (expandable)

RUN COMMAND:
    streamlit run src/app.py

DESIGN PHILOSOPHY:
    A dashboard is not just a display — it's a TEACHING TOOL.
    Every number shown has a label explaining what it means.
    Every colour is semantic: green = bullish, red = bearish, grey = neutral.
    An investor with zero technical knowledge should be able to read
    this dashboard and understand WHY a stock is strong or weak.

DEPENDENCIES:
    pip install streamlit matplotlib pandas numpy yfinance scikit-learn
"""

import sys
import os

# ── ensure src/ is on the path when running from project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings("ignore")

from data_fetch  import fetch_stock_data, SUPPORTED_STOCKS
from indicators  import add_all_indicators, get_latest_indicator_snapshot
from rules       import analyze_stock, StockSignal
from model       import run_ml_pipeline


# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Stock Analysis Engine — Indian Markets",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════
#  THEME & STYLE
# ══════════════════════════════════════════════════════════════════════

COLORS = {
    "bull":       "#00c896",   # emerald green
    "bear":       "#ff4b4b",   # vivid red
    "neutral":    "#a0aec0",   # slate grey
    "accent":     "#7c6af7",   # soft purple
    "gold":       "#f6c90e",   # signal gold
    "bg_dark":    "#0e1117",   # streamlit default dark
    "bg_card":    "#1a1f2e",   # card background
    "text":       "#e2e8f0",   # primary text
    "subtext":    "#718096",   # secondary text
    "up_candle":  "#26a69a",   # teal up candles
    "dn_candle":  "#ef5350",   # red down candles
}

CUSTOM_CSS = f"""
<style>
/* ── Overall Page ── */
.main .block-container {{ padding-top: 1.5rem; }}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {{
    background: {COLORS['bg_card']};
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 14px 18px;
}}

/* ── Signal Tag Chips ── */
.signal-chip {{
    display: inline-block;
    background: #2d3748;
    border-radius: 6px;
    padding: 4px 10px;
    margin: 3px 4px 3px 0;
    font-size: 0.82rem;
    font-weight: 600;
    color: {COLORS['text']};
    border: 1px solid #4a5568;
}}

/* ── Analyst Box ── */
.analyst-box {{
    background: {COLORS['bg_card']};
    border-left: 4px solid {COLORS['accent']};
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: {COLORS['text']};
}}

/* ── Section Headers ── */
.section-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLORS['text']};
    border-bottom: 1px solid #2d3748;
    padding-bottom: 6px;
    margin-bottom: 12px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}

/* ── ML Prediction Badge ── */
.pred-bull {{ color: {COLORS['bull']}; font-weight: 800; font-size: 1.6rem; }}
.pred-bear {{ color: {COLORS['bear']}; font-weight: 800; font-size: 1.6rem; }}
.pred-side {{ color: {COLORS['neutral']}; font-weight: 800; font-size: 1.6rem; }}

/* ── Info Box ── */
.info-box {{
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: {COLORS['subtext']};
    margin-top: 8px;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADER
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)   # cache 5 minutes
def load_and_analyze(ticker: str) -> dict:
    """
    Full pipeline: fetch → indicators → rules → ML.
    Cached so re-runs don't refetch data.
    """
    df_raw      = fetch_stock_data(ticker)
    df_enriched = add_all_indicators(df_raw)
    snap        = get_latest_indicator_snapshot(df_enriched)
    company     = df_raw["Company"].iloc[-1]

    signal      = analyze_stock(df_enriched, snap, ticker=ticker, company=company)
    ml_result   = run_ml_pipeline(df_enriched, model_type="random_forest")

    return {
        "df":        df_enriched,
        "snap":      snap,
        "signal":    signal,
        "ml":        ml_result,
        "ticker":    ticker,
        "company":   company,
    }


# ══════════════════════════════════════════════════════════════════════
#  CHART: CANDLESTICK + INDICATORS
# ══════════════════════════════════════════════════════════════════════

def plot_main_chart(df: pd.DataFrame, signal: StockSignal, days: int = 120) -> plt.Figure:
    """
    4-panel chart: Candlestick + MA + BB / RSI / MACD / Volume

    Uses matplotlib with dark theme to match dashboard aesthetic.
    """

    plot_df = df.tail(days).copy()
    dates   = np.arange(len(plot_df))
    idx     = plot_df.index

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs  = gridspec.GridSpec(4, 1, height_ratios=[4, 1.5, 1.5, 1.5],
                            hspace=0.06, figure=fig)

    ax1 = fig.add_subplot(gs[0])   # candlestick + MAs + BB
    ax2 = fig.add_subplot(gs[1])   # RSI
    ax3 = fig.add_subplot(gs[2])   # MACD
    ax4 = fig.add_subplot(gs[3])   # Volume

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#718096", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#2d3748")

    # ── Panel 1: Candlestick ──────────────────────────────────────
    for i, (_, row) in enumerate(plot_df.iterrows()):
        op, hi, lo, cl = row["Open"], row["High"], row["Low"], row["Close"]
        color  = COLORS["up_candle"] if cl >= op else COLORS["dn_candle"]
        body_h = abs(cl - op) or 0.01
        body_y = min(op, cl)

        ax1.add_patch(plt.Rectangle(
            (i - 0.35, body_y), 0.7, body_h,
            color=color, zorder=3
        ))
        ax1.plot([i, i], [lo, hi], color=color, linewidth=0.9, zorder=2)

    # Bollinger Bands
    if "BB_Upper" in plot_df.columns:
        ax1.fill_between(dates, plot_df["BB_Upper"], plot_df["BB_Lower"],
                         alpha=0.06, color=COLORS["accent"], label="BB Bands")
        ax1.plot(dates, plot_df["BB_Upper"], color=COLORS["accent"],
                 linewidth=0.6, linestyle="--", alpha=0.5)
        ax1.plot(dates, plot_df["BB_Lower"], color=COLORS["accent"],
                 linewidth=0.6, linestyle="--", alpha=0.5)

    # Moving Averages
    if "SMA_20" in plot_df.columns:
        ax1.plot(dates, plot_df["SMA_20"], color="#f6c90e",
                 linewidth=1.3, label="SMA 20", zorder=4)
    if "SMA_50" in plot_df.columns:
        ax1.plot(dates, plot_df["SMA_50"], color="#63b3ed",
                 linewidth=1.3, label="SMA 50", zorder=4)

    # Support / Resistance lines
    ax1.axhline(signal.support,    color=COLORS["bull"], linewidth=0.9,
                linestyle=":", alpha=0.8, label=f"Support ₹{signal.support:,.0f}")
    ax1.axhline(signal.resistance, color=COLORS["bear"], linewidth=0.9,
                linestyle=":", alpha=0.8, label=f"Resist ₹{signal.resistance:,.0f}")

    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.2,
               facecolor="#1a1f2e", edgecolor="#2d3748")
    ax1.set_ylabel("Price (₹)", color="#718096", fontsize=8)
    ax1.set_xlim(-1, len(dates))
    ax1.set_title(
        f"{signal.company} ({signal.ticker})  —  "
        f"Trend: {signal.trend} ({signal.trend_strength})  |  "
        f"Momentum: {signal.momentum_label}",
        color="#e2e8f0", fontsize=10, pad=10, fontweight="bold"
    )
    ax1.tick_params(labelbottom=False)

    # ── Panel 2: RSI ──────────────────────────────────────────────
    if "RSI" in plot_df.columns:
        rsi_vals = plot_df["RSI"].values
        ax2.plot(dates, rsi_vals, color=COLORS["gold"], linewidth=1.2, label="RSI")
        ax2.axhline(70, color=COLORS["bear"],    linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(30, color=COLORS["bull"],    linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(50, color=COLORS["neutral"], linewidth=0.5, linestyle=":", alpha=0.4)
        ax2.fill_between(dates, rsi_vals, 70,
                         where=(rsi_vals > 70), alpha=0.15, color=COLORS["bear"])
        ax2.fill_between(dates, rsi_vals, 30,
                         where=(rsi_vals < 30), alpha=0.15, color=COLORS["bull"])
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color="#718096", fontsize=8)
        ax2.text(len(dates) - 1, rsi_vals[-1], f" {rsi_vals[-1]:.1f}",
                 color=COLORS["gold"], fontsize=7.5, va="center")
    ax2.tick_params(labelbottom=False)
    ax2.set_xlim(-1, len(dates))

    # ── Panel 3: MACD ─────────────────────────────────────────────
    if "MACD" in plot_df.columns and "MACD_Signal" in plot_df.columns:
        macd_v  = plot_df["MACD"].values
        sig_v   = plot_df["MACD_Signal"].values
        hist_v  = plot_df["MACD_Hist"].values

        ax3.plot(dates, macd_v,  color="#63b3ed", linewidth=1.1, label="MACD")
        ax3.plot(dates, sig_v,   color="#fc8181", linewidth=1.1, label="Signal")
        ax3.axhline(0, color="#4a5568", linewidth=0.5)

        bar_colors = [COLORS["bull"] if h >= 0 else COLORS["bear"] for h in hist_v]
        ax3.bar(dates, hist_v, color=bar_colors, alpha=0.55, width=0.7, label="Histogram")
        ax3.legend(loc="upper left", fontsize=7, framealpha=0.2,
                   facecolor="#1a1f2e", edgecolor="#2d3748")
        ax3.set_ylabel("MACD", color="#718096", fontsize=8)
    ax3.tick_params(labelbottom=False)
    ax3.set_xlim(-1, len(dates))

    # ── Panel 4: Volume ───────────────────────────────────────────
    vol    = plot_df["Volume"].values
    closes = plot_df["Close"].values
    opens  = plot_df["Open"].values
    v_cols = [COLORS["up_candle"] if closes[i] >= opens[i]
              else COLORS["dn_candle"] for i in range(len(vol))]

    ax4.bar(dates, vol, color=v_cols, alpha=0.7, width=0.8)
    if "Vol_SMA" in plot_df.columns:
        ax4.plot(dates, plot_df["Vol_SMA"].values,
                 color=COLORS["gold"], linewidth=1.0,
                 linestyle="--", alpha=0.8, label="Vol MA20")
        ax4.legend(loc="upper left", fontsize=7, framealpha=0.2,
                   facecolor="#1a1f2e", edgecolor="#2d3748")

    ax4.set_ylabel("Volume", color="#718096", fontsize=8)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    # ── Shared X-axis labels ──────────────────────────────────────
    tick_step  = max(1, len(dates) // 8)
    tick_locs  = dates[::tick_step]
    tick_labels = [idx[i].strftime("%d %b") for i in range(0, len(idx), tick_step)]
    ax4.set_xticks(tick_locs)
    ax4.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7.5)
    ax4.set_xlim(-1, len(dates))

    fig.patch.set_facecolor("#0e1117")
    return fig


# ══════════════════════════════════════════════════════════════════════
#  CHART: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(importances: dict) -> plt.Figure:
    """Horizontal bar chart of Random Forest feature importances."""

    if not importances:
        return None

    items   = list(importances.items())[:8]   # top 8
    labels  = [k.replace("_", " ") for k, _ in items]
    values  = [v for _, v in items]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    bar_colors = [COLORS["accent"]] * len(values)
    bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1],
                   height=0.6, edgecolor="#2d3748")

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", ha="left",
                color="#e2e8f0", fontsize=8)

    ax.set_xlabel("Importance (%)", color="#718096", fontsize=8)
    ax.set_title("ML Feature Importance — What the Model Learned",
                 color="#e2e8f0", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#2d3748")
    ax.tick_params(colors="#718096", labelsize=8)
    ax.set_xlim(0, max(values) * 1.25)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════

def trend_color(trend: str) -> str:
    if trend == "Uptrend":   return COLORS["bull"]
    if trend == "Downtrend": return COLORS["bear"]
    return COLORS["neutral"]

def momentum_color(label: str) -> str:
    if "Bull" in label: return COLORS["bull"]
    if "Bear" in label: return COLORS["bear"]
    return COLORS["neutral"]

def pred_class(pred: str) -> str:
    if pred == "Bullish":  return "pred-bull"
    if pred == "Bearish":  return "pred-bear"
    return "pred-side"

def render_signal_chips(tags: list):
    chips_html = "".join(f'<span class="signal-chip">{t}</span>' for t in tags)
    st.markdown(chips_html, unsafe_allow_html=True)

def render_prob_bar(label: str, prob: float, color: str):
    pct = int(prob * 100)
    st.markdown(f"""
    <div style="margin: 5px 0;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.82rem; color:#a0aec0; margin-bottom:3px;">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div style="background:#2d3748; border-radius:4px; height:8px;">
            <div style="background:{color}; width:{pct}%;
                        border-radius:4px; height:8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def bool_badge(val: bool, true_label: str = "Yes", false_label: str = "No") -> str:
    if val:
        return f'<span style="color:{COLORS["bull"]}; font-weight:700;">✅ {true_label}</span>'
    return f'<span style="color:{COLORS["neutral"]};">— {false_label}</span>'


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## 📈 Stock Analysis Engine")
        st.markdown('<div class="info-box">Indian NSE Market · Powered by Yahoo Finance + Random Forest ML</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        ticker = st.selectbox(
            "🏢 Select Stock",
            options=list(SUPPORTED_STOCKS.keys()),
            format_func=lambda t: f"{SUPPORTED_STOCKS[t]} ({t})",
        )

        days = st.slider("📅 Chart History (days)", 30, 365, 120, step=10)

        model_type = st.radio(
            "🤖 ML Model",
            ["random_forest", "logistic_regression"],
            format_func=lambda x: "Random Forest" if x == "random_forest" else "Logistic Regression",
        )

        st.markdown("---")
        refresh = st.button("🔄 Refresh Data", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <b>How to read this dashboard:</b><br><br>
        🟢 <b>Momentum Score</b> — counts how many indicators are aligned.<br><br>
        📊 <b>ML Prediction</b> — what the Random Forest learned from history.<br><br>
        ⚡ <b>Volume Breakout</b> — institutional money moving.<br><br>
        ⚠️ <b>Failed Breakout</b> — retail trap signal.<br><br>
        📋 <b>Analyst Note</b> — plain-English reasoning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("⚠️ Educational tool only. Not financial advice.")

    return ticker, days, model_type, refresh


# ══════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════

def render_dashboard(data: dict, days: int):
    signal: StockSignal = data["signal"]
    snap                = data["snap"]
    ml                  = data["ml"]
    df                  = data["df"]

    # ── Header ────────────────────────────────────────────────────
    close     = snap["close"] or 0
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else close
    day_chg   = close - prev_close
    day_pct   = (day_chg / prev_close * 100) if prev_close else 0
    chg_color = COLORS["bull"] if day_chg >= 0 else COLORS["bear"]
    arrow     = "▲" if day_chg >= 0 else "▼"

    st.markdown(f"""
    <div style="padding: 16px 0 8px 0;">
        <span style="font-size:1.9rem; font-weight:800; color:#e2e8f0;">
            {signal.company}
        </span>
        <span style="font-size:1.1rem; color:#718096; margin-left:10px;">
            {signal.ticker}
        </span>
        <span style="font-size:1.6rem; font-weight:700; color:#e2e8f0; margin-left:20px;">
            ₹{close:,.2f}
        </span>
        <span style="font-size:1.1rem; font-weight:600; color:{chg_color}; margin-left:10px;">
            {arrow} ₹{abs(day_chg):.2f} ({day_pct:+.2f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPI Row ───────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("RSI", f"{snap['rsi']:.1f}" if snap['rsi'] else "—",
                  help="Relative Strength Index. >70=Overbought, <30=Oversold, 50–70=Healthy Bull.")
    with k2:
        st.metric("SMA 20", f"₹{snap['sma_20']:,.0f}" if snap['sma_20'] else "—",
                  delta=f"{'above' if snap['close'] > snap['sma_20'] else 'below'}",
                  help="20-day Simple Moving Average. Short-term trend indicator.")
    with k3:
        st.metric("SMA 50", f"₹{snap['sma_50']:,.0f}" if snap['sma_50'] else "—",
                  delta=f"{'above' if snap['close'] > snap['sma_50'] else 'below'}",
                  help="50-day Simple Moving Average. Medium-term trend indicator.")
    with k4:
        vr = snap["vol_ratio"] or 1.0
        st.metric("Volume Ratio", f"{vr:.2f}x",
                  delta="spike" if vr >= 1.5 else "normal",
                  help="Today's volume ÷ 20-day average. >1.5 = institutional activity.")
    with k5:
        st.metric("BB %B", f"{snap['bb_pctb']:.2f}" if snap['bb_pctb'] is not None else "—",
                  help="Position within Bollinger Bands. 0=lower band, 0.5=middle, 1=upper band.")
    with k6:
        st.metric("Momentum Score", f"{signal.momentum_score:+d} / +5",
                  help="Counts how many indicators agree on direction. +5=max bull, -5=max bear.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Chart ────────────────────────────────────────────────
    with st.spinner("Rendering chart..."):
        fig = plot_main_chart(df, signal, days=days)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Second Row: Signal Panel + ML Panel ──────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Signal Summary ──────────────────────────────────────
    with left:
        st.markdown('<div class="section-header">📊 Signal Summary</div>',
                    unsafe_allow_html=True)

        # Trend & Momentum
        t_color = trend_color(signal.trend)
        m_color = momentum_color(signal.momentum_label)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background:#1a1f2e; border-radius:10px; padding:14px;
                        border:1px solid #2d3748; text-align:center;">
                <div style="font-size:0.75rem; color:#718096; margin-bottom:4px;">TREND</div>
                <div style="font-size:1.4rem; font-weight:800; color:{t_color};">
                    {signal.trend}
                </div>
                <div style="font-size:0.8rem; color:#718096;">{signal.trend_strength}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#1a1f2e; border-radius:10px; padding:14px;
                        border:1px solid #2d3748; text-align:center;">
                <div style="font-size:0.75rem; color:#718096; margin-bottom:4px;">MOMENTUM</div>
                <div style="font-size:1.4rem; font-weight:800; color:{m_color};">
                    {signal.momentum_label}
                </div>
                <div style="font-size:0.8rem; color:#718096;">Score: {signal.momentum_score:+d}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Support & Resistance
        st.markdown('<div class="section-header">🏗️ Support & Resistance</div>',
                    unsafe_allow_html=True)
        sr1, sr2 = st.columns(2)
        with sr1:
            st.markdown(f"""
            <div style="background:#1a2035; border-radius:8px; padding:12px;
                        border-left:3px solid {COLORS['bull']};">
                <div style="font-size:0.72rem; color:#718096;">SUPPORT</div>
                <div style="font-size:1.2rem; font-weight:700; color:{COLORS['bull']};">
                    ₹{signal.support:,.2f}
                </div>
                <div style="font-size:0.72rem; color:#718096;">{signal.support_method}</div>
            </div>
            """, unsafe_allow_html=True)
        with sr2:
            st.markdown(f"""
            <div style="background:#1a2035; border-radius:8px; padding:12px;
                        border-left:3px solid {COLORS['bear']};">
                <div style="font-size:0.72rem; color:#718096;">RESISTANCE</div>
                <div style="font-size:1.2rem; font-weight:700; color:{COLORS['bear']};">
                    ₹{signal.resistance:,.2f}
                </div>
                <div style="font-size:0.72rem; color:#718096;">{signal.resistance_method}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Breakout Flags
        st.markdown('<div class="section-header">⚡ Breakout Analysis</div>',
                    unsafe_allow_html=True)
        bf1, bf2 = st.columns(2)
        with bf1:
            bo_label = f"Yes — {signal.breakout_direction}" if signal.volume_breakout else "No"
            bo_color = COLORS["bull"] if signal.volume_breakout and signal.breakout_direction == "Bullish" \
                       else COLORS["bear"] if signal.volume_breakout else COLORS["neutral"]
            st.markdown(f"""
            <div style="background:#1a2035; border-radius:8px; padding:12px;
                        border:1px solid #2d3748;">
                <div style="font-size:0.72rem; color:#718096; margin-bottom:4px;">
                    VOLUME BREAKOUT
                </div>
                <div style="font-size:0.95rem; font-weight:700; color:{bo_color};">
                    {bo_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with bf2:
            fb_label = f"Yes — {signal.failed_breakout_dir}" if signal.failed_breakout else "No"
            fb_color = COLORS["bear"] if signal.failed_breakout else COLORS["neutral"]
            st.markdown(f"""
            <div style="background:#1a2035; border-radius:8px; padding:12px;
                        border:1px solid #2d3748;">
                <div style="font-size:0.72rem; color:#718096; margin-bottom:4px;">
                    FAILED BREAKOUT
                </div>
                <div style="font-size:0.95rem; font-weight:700; color:{fb_color};">
                    {fb_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Individual Signal Flags
        st.markdown('<div class="section-header">🔬 Individual Signals</div>',
                    unsafe_allow_html=True)

        flags = {
            "Price > SMA 20":     signal.price_above_sma20,
            "Price > SMA 50":     signal.price_above_sma50,
            "MACD Bullish":       signal.macd_bullish,
            "MACD Hist Rising":   signal.macd_hist_rising,
            "RSI Bullish Zone":   signal.rsi_bullish_zone,
            "RSI Overbought":     signal.rsi_overbought,
            "RSI Oversold":       signal.rsi_oversold,
            "Volume Above Avg":   signal.vol_above_avg,
            "Golden Cross":       signal.golden_cross,
            "Death Cross":        signal.death_cross,
            "BB Squeeze":         signal.bb_squeeze,
            "BB Upper Touch":     signal.bb_upper_touch,
        }

        fc1, fc2 = st.columns(2)
        items = list(flags.items())
        for i, (name, val) in enumerate(items):
            col = fc1 if i % 2 == 0 else fc2
            icon = "🟢" if val else "⚫"
            color = "#00c896" if val else "#4a5568"
            col.markdown(
                f'<div style="font-size:0.82rem; color:{color}; '
                f'padding:2px 0;">{icon} {name}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Signal Tags
        st.markdown('<div class="section-header">🏷️ Signal Tags</div>',
                    unsafe_allow_html=True)
        render_signal_chips(signal.signal_tags)

    # ── RIGHT: ML Panel ───────────────────────────────────────────
    with right:
        st.markdown('<div class="section-header">🤖 ML Classification</div>',
                    unsafe_allow_html=True)

        if ml.get("trained"):
            pred       = ml.get("prediction", "Unknown")
            conf       = ml.get("confidence", 0.0)
            probs      = ml.get("probabilities", {})
            cv_acc     = ml.get("accuracy", 0.0)
            label_dist = ml.get("label_dist", {})

            # Prediction badge
            p_class = pred_class(pred)
            pred_color = COLORS["bull"] if pred == "Bullish" \
                         else COLORS["bear"] if pred == "Bearish" \
                         else COLORS["neutral"]

            st.markdown(f"""
            <div style="background:#1a1f2e; border-radius:12px; padding:20px;
                        border:1px solid #2d3748; text-align:center; margin-bottom:16px;">
                <div style="font-size:0.75rem; color:#718096; margin-bottom:6px;">
                    CURRENT STATE PREDICTION
                </div>
                <div style="font-size:2rem; font-weight:900; color:{pred_color};">
                    {pred.upper()}
                </div>
                <div style="font-size:0.85rem; color:#718096; margin-top:6px;">
                    Confidence: <b style="color:{pred_color};">{conf*100:.1f}%</b>
                </div>
                <div style="font-size:0.75rem; color:#4a5568; margin-top:4px;">
                    CV Accuracy: {cv_acc*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Bars
            st.markdown("**Class Probabilities:**")
            prob_order = [
                ("Bullish",  COLORS["bull"]),
                ("Sideways", COLORS["neutral"]),
                ("Bearish",  COLORS["bear"]),
            ]
            for cls, clr in prob_order:
                p = probs.get(cls, 0.0)
                render_prob_bar(cls, p, clr)

            st.markdown("<br>", unsafe_allow_html=True)

            # Label distribution
            st.markdown("**Training Data Distribution:**")
            total = sum(label_dist.values()) or 1
            for cls, cnt in label_dist.items():
                pct = cnt / total * 100
                clr = COLORS["bull"] if cls == "Bullish" \
                      else COLORS["bear"] if cls == "Bearish" \
                      else COLORS["neutral"]
                render_prob_bar(f"{cls} ({cnt} days)", pct / 100, clr)

            st.markdown("<br>", unsafe_allow_html=True)

            # Feature Importance chart
            imp = ml.get("importances", {})
            if imp:
                st.markdown('<div class="section-header">📐 Feature Importance</div>',
                            unsafe_allow_html=True)
                fig_imp = plot_feature_importance(imp)
                if fig_imp:
                    st.pyplot(fig_imp, use_container_width=True)
                    plt.close(fig_imp)

                st.markdown(f"""
                <div class="info-box">
                {ml.get("importance_text","").replace(chr(10),"<br>")}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning(f"ML model could not be trained: {ml.get('reason', 'Unknown error.')}")

    # ── Analyst Explanation ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Analyst Explanation</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="analyst-box">{signal.explanation}</div>',
                unsafe_allow_html=True)

    # ── Raw Data Table (expandable) ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🗃️ View Raw Data + All Indicators", expanded=False):
        display_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "SMA_20", "SMA_50",
            "BB_Upper", "BB_Lower", "BB_PctB", "BB_Width",
            "Vol_Ratio", "Vol_Spike",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].tail(60).copy()

        # colour Close column
        def colour_close(val):
            return f"color: {COLORS['bull']}" if val > 0 else f"color: {COLORS['bear']}"

        st.dataframe(
            display_df.style.format({
                "Open": "₹{:.2f}", "High": "₹{:.2f}", "Low": "₹{:.2f}",
                "Close": "₹{:.2f}", "Volume": "{:,.0f}",
                "RSI": "{:.1f}", "MACD": "{:.4f}", "MACD_Signal": "{:.4f}",
                "MACD_Hist": "{:.4f}", "SMA_20": "₹{:.2f}", "SMA_50": "₹{:.2f}",
                "BB_Upper": "₹{:.2f}", "BB_Lower": "₹{:.2f}",
                "BB_PctB": "{:.3f}", "BB_Width": "{:.4f}",
                "Vol_Ratio": "{:.2f}x", "Vol_Spike": "{:.0f}",
            }),
            use_container_width=True,
            height=400,
        )

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#4a5568; font-size:0.78rem; padding:20px 0;">
        Stock Analysis Engine · Indian NSE Markets · Built with Python, Streamlit & scikit-learn<br>
        ⚠️ This tool is for <b>educational purposes only</b> and does not constitute financial advice.
        Past patterns do not guarantee future results. Always do your own research.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    ticker, days, model_type, refresh = render_sidebar()

    if refresh:
        st.cache_data.clear()

    with st.spinner(f"Fetching and analysing {SUPPORTED_STOCKS.get(ticker, ticker)}..."):
        try:
            data = load_and_analyze(ticker)
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            st.info("Check your internet connection and try again.")
            return

    render_dashboard(data, days=days)


if __name__ == "__main__":
    main()
