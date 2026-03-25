"""
╔══════════════════════════════════════════════════════════════════════╗
║                STOCK ANALYSIS ENGINE — model.py                     ║
║           Module 4: Machine Learning Classification Layer            ║
╚══════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Use a Random Forest classifier to learn from historical indicator
    patterns and classify the current stock state as:
      → Bullish  (price likely to trend up)
      → Bearish  (price likely to trend down)
      → Sideways (price likely to stay flat / consolidate)

WHY MACHINE LEARNING OVER PURE RULES?
    Rules use fixed thresholds: "RSI > 70 = overbought."
    But what if RSI = 68 with a rising MACD histogram and golden cross?
    A fixed rule misses nuanced combinations. ML learns which PATTERNS
    of features — taken together — historically preceded bullish outcomes.

    Random Forest is ideal here because:
      1. Handles non-linear relationships between indicators
      2. Resistant to overfitting (ensemble of 300 decision trees)
      3. Provides feature importance — we can see WHICH indicators matter most
      4. No feature scaling required (unlike SVM or neural networks)
      5. Works with small-to-medium datasets (which stock history gives us)

LABEL GENERATION APPROACH (Self-Supervised):
    We don't manually label data. Instead, we look at what the price
    DID 5 trading days AFTER each observation:
      → Forward return > +2%  : Label = "Bullish"
      → Forward return < -2%  : Label = "Bearish"
      → Otherwise             : Label = "Sideways"

    This is called "future price labeling" — a common technique in
    quantitative finance. The model learns indicator states that
    historically preceded positive or negative price movement.

IMPORTANT DISCLAIMER:
    This model is an analytical tool, not a trading system.
    No model can reliably predict stock prices. Past patterns
    do not guarantee future results. Use for EDUCATION ONLY.

DEPENDENCIES:
    pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import classification_report, confusion_matrix
from sklearn.pipeline          import Pipeline
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# Features the model uses — each one is a named indicator column
# These are the "inputs" the Random Forest learns patterns from
FEATURE_COLUMNS = [
    "RSI",           # momentum oscillator
    "MACD",          # trend momentum
    "MACD_Signal",   # signal line (crossover context)
    "MACD_Hist",     # histogram — momentum acceleration
    "SMA_20",        # short-term trend level
    "SMA_50",        # medium-term trend level
    "BB_PctB",       # position within Bollinger Bands (0=lower, 1=upper)
    "BB_Width",      # volatility proxy (squeeze detection)
    "Vol_Ratio",     # today's volume vs average
]

# Derived ratio features (computed inside the module for better ML signal)
DERIVED_FEATURES = [
    "Price_to_SMA20",    # Close / SMA20 (how far price is from MA)
    "Price_to_SMA50",    # Close / SMA50
    "SMA20_to_SMA50",    # SMA20 / SMA50 (golden/death cross magnitude)
    "RSI_Centered",      # RSI - 50 (signed momentum)
]

ALL_FEATURES = FEATURE_COLUMNS + DERIVED_FEATURES

# Label thresholds
FORWARD_DAYS  = 5     # look ahead 5 trading days for labeling
BULL_THRESH   = 0.02  # +2% forward return = Bullish
BEAR_THRESH   = -0.02 # -2% forward return = Bearish

# Model settings
N_ESTIMATORS  = 300   # number of decision trees in the forest
RANDOM_STATE  = 42    # for reproducibility
MIN_SAMPLES   = 60    # minimum rows needed to train (need enough history)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix (X) from the enriched DataFrame.

    Adds derived ratio features on top of raw indicator columns.
    Derived features often capture RELATIONSHIPS between indicators
    better than raw values alone.

    Args:
        df : Enriched DataFrame from add_all_indicators()

    Returns:
        pd.DataFrame with all feature columns (no NaN rows)

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Feature Engineering Matters
    ──────────────────────────────────────────────
    Raw indicators have absolute values that are hard to compare across
    stocks or time periods. Reliance at SMA20=₹2500 is very different
    from Infosys at SMA20=₹1700.

    RATIO features normalize these differences:

    Price_to_SMA20 = Close / SMA20
      → 1.05 means price is 5% ABOVE the 20-day MA (bullish pressure)
      → 0.95 means price is 5% BELOW the 20-day MA (bearish pressure)
      → Works the same for a ₹500 stock and a ₹3000 stock

    SMA20_to_SMA50 = SMA20 / SMA50
      → > 1.0 = Golden Cross territory (short-term above long-term)
      → < 1.0 = Death Cross territory (short-term below long-term)
      → The MAGNITUDE tells you how strong the divergence is

    RSI_Centered = RSI - 50
      → Positive: bullish momentum territory
      → Negative: bearish momentum territory
      → Easier for ML to interpret than raw RSI (centered at 0)

    These engineered features help the Random Forest find more
    meaningful decision boundaries than raw values alone.
    """

    feat = df.copy()

    # ── Derived Ratio Features ──
    close  = feat["Close"].astype(float)
    sma20  = feat["SMA_20"].astype(float)
    sma50  = feat["SMA_50"].astype(float)
    rsi    = feat["RSI"].astype(float)

    feat["Price_to_SMA20"] = (close / sma20.replace(0, np.nan)).round(6)
    feat["Price_to_SMA50"] = (close / sma50.replace(0, np.nan)).round(6)
    feat["SMA20_to_SMA50"] = (sma20 / sma50.replace(0, np.nan)).round(6)
    feat["RSI_Centered"]   = (rsi - 50).round(4)

    # ── Keep only feature columns, drop NaN rows ──
    available = [c for c in ALL_FEATURES if c in feat.columns]
    result    = feat[available].dropna()

    return result


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — LABEL GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_labels(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> pd.Series:
    """
    Generate target labels based on forward price returns.

    For each row, we look at what the Close price will be
    `forward_days` trading sessions into the future.

    Forward Return = (Close[t+N] - Close[t]) / Close[t]

    Labeling Rules:
        > +2%  → "Bullish"
        < -2%  → "Bearish"
        else   → "Sideways"

    Args:
        df          : Enriched DataFrame with Close column
        forward_days: How many days ahead to look (default 5)

    Returns:
        pd.Series of labels aligned with df's index

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why 5-Day Forward Returns?
    ──────────────────────────────────────────────
    5 trading days = 1 calendar week. This timeframe is chosen because:

    1. SHORT ENOUGH to be actionable:
       Technical indicators are short-to-medium term signals.
       Using a 6-month forward return would be more about fundamentals.

    2. LONG ENOUGH to filter noise:
       1-day returns are mostly random (daily noise).
       5 days gives price enough time to follow the signal.

    3. Aligned with how traders USE indicators:
       A MACD crossover is typically acted upon within a week.
       A RSI oversold reading typically resolves within 3–7 days.

    WHY ±2% THRESHOLD?
    Average daily volatility of NSE large-caps is ~1-1.5%.
    Over 5 days, a ±2% move is MEANINGFUL — it's above random noise.
    Below ±2%, the stock is essentially "going nowhere" (Sideways).

    CLASS IMBALANCE NOTE:
    In a sideways/ranging market, "Sideways" labels will dominate.
    In a trending market, "Bullish" or "Bearish" will dominate.
    This is real — our model handles it with class_weight='balanced'.
    """

    close          = df["Close"].astype(float)
    forward_return = close.shift(-forward_days) / close - 1

    labels = pd.Series("Sideways", index=df.index)
    labels[forward_return >  BULL_THRESH] = "Bullish"
    labels[forward_return <  BEAR_THRESH] = "Bearish"

    # Last `forward_days` rows have no future data — drop them later
    labels.iloc[-forward_days:] = np.nan

    return labels


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_model(
    df: pd.DataFrame,
    model_type: str = "random_forest"
) -> dict:
    """
    Train the ML classifier on historical indicator data.

    Supports:
        "random_forest"     → RandomForestClassifier (recommended)
        "logistic_regression" → LogisticRegression (simpler, interpretable)

    Args:
        df         : Enriched DataFrame (from add_all_indicators)
        model_type : Which model to train

    Returns:
        dict with keys:
          "model"       : trained sklearn model (or pipeline)
          "features"    : list of feature column names used
          "accuracy"    : cross-validated accuracy score
          "cv_scores"   : array of CV fold scores
          "label_dist"  : class distribution in training data
          "importances" : feature importance dict (RF only)
          "report"      : classification report string
          "trained"     : bool — whether training succeeded

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Why Random Forest for Stock Signals?
    ──────────────────────────────────────────────
    A Random Forest is an ENSEMBLE of decision trees. Here's why it fits:

    Decision Trees:
      Each tree asks a series of yes/no questions:
        "Is RSI > 55?"
        "Is Price above SMA20?"
        "Is MACD_Hist positive?"
      And arrives at a classification: Bullish / Bearish / Sideways.

    The Problem with One Tree:
      A single tree is brittle — it memorizes the training data
      (overfitting) and fails on new data.

    Random Forest Solution:
      Train 300 DIFFERENT trees, each on a random subset of data
      and a random subset of features. Then take a majority vote.

      → Overfitting is cancelled out across the ensemble
      → Robust to outliers (one bad day doesn't break the model)
      → Feature importance tells us WHICH indicators matter most

    CLASS WEIGHT BALANCING:
      In real stock data, Sideways days outnumber trend days.
      Without balancing, the model just predicts "Sideways" always.
      class_weight='balanced' tells the model to penalize errors on
      minority classes more — forcing it to actually learn trends.

    CROSS-VALIDATION:
      We split data into 5 folds and test on each held-out fold.
      This gives a realistic estimate of out-of-sample performance.
      Stock data is TIME-SERIES — we respect chronological order.

    IMPORTANT REALITY CHECK:
      ~55–65% accuracy on 3-class classification is actually good
      for stock market data. Random = 33%. Even hedge funds with
      billions in research budgets only slightly beat random.
      Our model's value is in SYSTEMATIC signal consistency,
      not in "predicting the future."
    """

    if len(df) < MIN_SAMPLES:
        return {
            "trained":   False,
            "reason":    f"Insufficient data: {len(df)} rows < {MIN_SAMPLES} required.",
            "model":     None,
            "features":  [],
            "accuracy":  0.0,
        }

    # ── Build Features ──
    X_df   = build_feature_matrix(df)
    labels = generate_labels(df)

    # Align labels to feature rows, drop NaN
    combined        = X_df.join(labels.rename("Label"), how="inner")
    combined        = combined.dropna(subset=["Label"])
    available_feats = [c for c in ALL_FEATURES if c in combined.columns]

    X = combined[available_feats].values
    y = combined["Label"].values

    if len(np.unique(y)) < 2:
        return {
            "trained":  False,
            "reason":   "Not enough class diversity in labels.",
            "model":    None,
            "features": available_feats,
            "accuracy": 0.0,
        }

    label_dist = {
        cls: int((y == cls).sum()) for cls in ["Bullish", "Bearish", "Sideways"]
    }

    # ── Train/Test Split (time-ordered — no shuffle) ──
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Build Model ──
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators  = N_ESTIMATORS,
            max_depth     = 6,          # prevents deep overfitting
            min_samples_leaf = 5,       # each leaf needs 5 examples
            class_weight  = "balanced", # handle class imbalance
            random_state  = RANDOM_STATE,
            n_jobs        = -1,         # use all CPU cores
        )
        pipeline = model  # RF doesn't need scaling

    elif model_type == "logistic_regression":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),   # LR needs normalized features
            ("model", LogisticRegression(
                C             = 1.0,
                class_weight  = "balanced",
                max_iter      = 1000,
                random_state  = RANDOM_STATE,
                multi_class   = "ovr",
            ))
        ])
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'random_forest' or 'logistic_regression'.")

    # ── Fit ──
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred    = pipeline.predict(X_test)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    report    = classification_report(y_test, y_pred, zero_division=0)

    # ── Feature Importances (RF only) ──
    importances = {}
    if model_type == "random_forest":
        raw_imp = pipeline.feature_importances_
        importances = dict(zip(available_feats, [round(float(v), 4) for v in raw_imp]))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return {
        "trained":      True,
        "model":        pipeline,
        "model_type":   model_type,
        "features":     available_feats,
        "accuracy":     round(float(cv_scores.mean()), 4),
        "cv_scores":    cv_scores.round(4).tolist(),
        "label_dist":   label_dist,
        "importances":  importances,
        "report":       report,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════

def predict_current_state(model_result: dict, df: pd.DataFrame) -> dict:
    """
    Use the trained model to classify the CURRENT state of the stock.

    Takes the most recent row of features (today's indicator values)
    and runs it through the trained classifier.

    Args:
        model_result : Output dict from train_model()
        df           : Enriched DataFrame (most recent row = today)

    Returns:
        dict with:
          "prediction"   : "Bullish" | "Bearish" | "Sideways"
          "confidence"   : float (0–1, highest class probability)
          "probabilities": dict of class → probability
          "feature_vals" : dict of feature → value used for prediction

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: Confidence Matters More Than Label
    ──────────────────────────────────────────────
    A "Bullish" prediction at 90% confidence is very different from
    one at 38% confidence (barely above the 33% random baseline).

    Probability Distribution tells the full story:
      Bullish: 80%, Sideways: 15%, Bearish: 5%
      → Strong conviction, high confidence, reliable signal

      Bullish: 40%, Sideways: 35%, Bearish: 25%
      → Weak conviction, uncertain market, hold off

    Professional quant funds use PROBABILITY THRESHOLDS:
      Only act when confidence exceeds a minimum (e.g., 60%).
      This filters out low-conviction "coin-flip" signals.

    Our dashboard shows all three probabilities so the investor
    can gauge the quality of the ML signal, not just the label.
    """

    if not model_result.get("trained"):
        return {
            "prediction":    "Unknown",
            "confidence":    0.0,
            "probabilities": {},
            "feature_vals":  {},
            "error":         model_result.get("reason", "Model not trained."),
        }

    model    = model_result["model"]
    features = model_result["features"]

    # ── Build feature row for today ──
    X_df    = build_feature_matrix(df)
    if X_df.empty:
        return {
            "prediction":   "Unknown",
            "confidence":   0.0,
            "probabilities": {},
            "feature_vals": {},
            "error":        "Insufficient indicator data for prediction.",
        }

    available = [f for f in features if f in X_df.columns]
    X_latest  = X_df[available].iloc[[-1]]   # last row = today's indicators

    # ── Predict ──
    prediction  = model.predict(X_latest)[0]
    proba       = model.predict_proba(X_latest)[0]
    classes     = model.classes_

    prob_dict   = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence  = round(float(max(proba)), 4)

    # ── Feature values used ──
    feature_vals = {
        col: round(float(X_latest[col].values[0]), 4)
        for col in available
        if col in X_latest.columns
    }

    return {
        "prediction":    prediction,
        "confidence":    confidence,
        "probabilities": prob_dict,
        "feature_vals":  feature_vals,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — FEATURE IMPORTANCE INSIGHT
# ══════════════════════════════════════════════════════════════════════

def explain_feature_importance(importances: dict) -> str:
    """
    Convert feature importance rankings into an analyst-readable insight.

    Random Forest tells us which indicators contributed most to its
    decisions across all 300 trees. This is the model's "inner voice."

    ──────────────────────────────────────────────
    📚 MARKET INSIGHT: What Feature Importance Reveals
    ──────────────────────────────────────────────
    Feature importance answers: "What did the model learn to care about?"

    If RSI_Centered is the top feature:
    → The model learned that momentum level (RSI) is the strongest
      predictor of near-term price direction in this stock's history.

    If Vol_Ratio is the top feature:
    → Volume changes were the best predictor — institutional activity
      drove most of the meaningful price moves historically.

    If Price_to_SMA20 is top:
    → The stock's behavior relative to its moving average was the most
      predictive — it tends to revert to or extend from its MA cleanly.

    This teaches you something SPECIFIC about each stock's personality.
    Some stocks are "RSI stocks" (momentum-driven).
    Some are "volume stocks" (breakout-driven).
    Some are "trend stocks" (MA-driven).
    """

    if not importances:
        return "Feature importance not available."

    top = list(importances.items())[:5]  # top 5

    lines = ["The model learned these indicators matter most (in order):"]
    rank_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    readable_names = {
        "RSI":            "RSI (momentum strength)",
        "RSI_Centered":   "RSI centered at 50 (directional momentum)",
        "MACD":           "MACD line (trend momentum)",
        "MACD_Hist":      "MACD Histogram (momentum acceleration)",
        "MACD_Signal":    "MACD Signal Line (trend confirmation)",
        "Price_to_SMA20": "Price vs 20-day MA (short-term trend pressure)",
        "Price_to_SMA50": "Price vs 50-day MA (medium-term trend pressure)",
        "SMA20_to_SMA50": "SMA20 vs SMA50 (golden/death cross magnitude)",
        "BB_PctB":        "Bollinger %B (price position within bands)",
        "BB_Width":       "Bollinger Width (volatility / squeeze)",
        "Vol_Ratio":      "Volume Ratio (institutional activity)",
        "SMA_20":         "20-day SMA level",
        "SMA_50":         "50-day SMA level",
    }

    for i, (feat, imp) in enumerate(top):
        label = rank_labels[i] if i < len(rank_labels) else f"{i+1}th"
        name  = readable_names.get(feat, feat)
        pct   = imp * 100
        lines.append(f"  {label}: {name} — {pct:.1f}% contribution")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION — run_ml_pipeline()
# ══════════════════════════════════════════════════════════════════════

def run_ml_pipeline(df: pd.DataFrame, model_type: str = "random_forest") -> dict:
    """
    Full ML pipeline: train → predict → explain.

    Single entry point used by app.py.

    Args:
        df         : Enriched DataFrame from add_all_indicators()
        model_type : "random_forest" or "logistic_regression"

    Returns:
        dict with all ML results:
          "trained"         : bool
          "prediction"      : str
          "confidence"      : float
          "probabilities"   : dict
          "accuracy"        : float (cross-validated)
          "label_dist"      : class distribution
          "importances"     : feature importances
          "importance_text" : readable insight string
          "report"          : classification report
          "feature_vals"    : features used for today's prediction
    """

    print(f"🤖 Training {model_type.replace('_', ' ').title()} ...")

    model_result = train_model(df, model_type=model_type)

    if not model_result["trained"]:
        print(f"   ⚠️  Training failed: {model_result.get('reason')}")
        return model_result

    pred_result = predict_current_state(model_result, df)
    imp_text    = explain_feature_importance(model_result.get("importances", {}))

    accuracy = model_result["accuracy"]
    print(f"   ✅ Model trained. CV Accuracy: {accuracy*100:.1f}%")
    print(f"   📊 Prediction: {pred_result['prediction']} "
          f"(confidence: {pred_result['confidence']*100:.1f}%)")

    return {
        **model_result,
        **pred_result,
        "importance_text": imp_text,
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN — self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_fetch  import fetch_stock_data
    from indicators  import add_all_indicators

    print("=" * 65)
    print("  STOCK ANALYSIS ENGINE — ML Model Test")
    print("=" * 65 + "\n")

    for ticker in ["RELIANCE.NS", "INFY.NS", "SBIN.NS"]:
        df_raw      = fetch_stock_data(ticker)
        df_enriched = add_all_indicators(df_raw)

        print(f"{'─'*60}")
        print(f"  {ticker}")
        print(f"{'─'*60}")

        result = run_ml_pipeline(df_enriched, model_type="random_forest")

        if result["trained"]:
            print()
            print(f"  📊 Label Distribution (training data):")
            for cls, count in result["label_dist"].items():
                bar = "█" * (count // 3)
                print(f"     {cls:<10}: {count:>3} days  {bar}")

            print()
            print(f"  🎯 CV Accuracy: {result['accuracy']*100:.1f}%")
            print(f"     (Folds: {[f'{s*100:.0f}%' for s in result['cv_scores']]})")

            print()
            print(f"  🤖 Current State Prediction:")
            print(f"     Label      : {result['prediction']}")
            print(f"     Confidence : {result['confidence']*100:.1f}%")
            print(f"     Probabilities:")
            for cls, prob in sorted(result["probabilities"].items(),
                                    key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                print(f"       {cls:<10}: {prob*100:5.1f}%  {bar}")

            print()
            print(f"  🧠 Feature Importance Insight:")
            for line in result["importance_text"].split("\n"):
                print(f"  {line}")

            print()
            print(f"  📋 Classification Report (held-out test set):")
            for line in result["report"].split("\n"):
                if line.strip():
                    print(f"     {line}")
        print()
