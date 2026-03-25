# 📈 Stock Analysis Engine — Indian Markets

An intelligent stock analysis system for NSE Indian stocks. Combines
technical analysis, market intelligence rules, and machine learning to
help investors understand **why** a stock is strong or weak — not just
whether to buy or sell.

---

## 🗂️ Project Structure

```
stock_analysis_engine/
├── requirements.txt
└── src/
    ├── data_fetch.py    # Module 1 — Fetch OHLCV data from Yahoo Finance
    ├── indicators.py    # Module 2 — RSI, MACD, SMA, Bollinger Bands, Volume MA
    ├── rules.py         # Module 3 — Trend detection, breakout logic, explanation
    ├── model.py         # Module 4 — Random Forest ML classifier
    └── app.py           # Module 5 — Streamlit dashboard (run this)
```

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
streamlit run src/app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 🧪 Test Individual Modules

```bash
# Test data fetching
python src/data_fetch.py

# Test indicators
python src/indicators.py

# Test rules engine
python src/rules.py

# Test ML model
python src/model.py
```

---

## 📊 Supported Stocks

| Ticker        | Company              |
|---------------|----------------------|
| RELIANCE.NS   | Reliance Industries  |
| INFY.NS       | Infosys              |
| SBIN.NS       | State Bank of India  |

---

## 🔬 Features

### Technical Indicators
- **RSI** — Momentum oscillator (overbought / oversold)
- **MACD** — Trend momentum + crossover signals
- **SMA 20 / SMA 50** — Short and medium-term trend direction
- **Bollinger Bands** — Volatility + squeeze detection
- **Volume MA** — Institutional activity detection

### Market Intelligence
- Uptrend / Downtrend / Sideways classification (3-layer logic)
- Support & Resistance from swing highs/lows
- Volume Breakout detection
- Failed Breakout (bull/bear trap) detection
- Momentum scoring (-5 to +5)
- Analyst-style explanation generation

### Machine Learning
- Random Forest (300 trees) OR Logistic Regression
- Self-supervised labels from 5-day forward returns
- Feature importance — discovers which indicators matter most per stock
- Prediction with probability confidence scores
- Cross-validated accuracy reporting

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It does not constitute financial advice.
Past indicator patterns do not guarantee future price performance.
Always do your own research before making investment decisions.
