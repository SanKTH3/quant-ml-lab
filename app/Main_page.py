from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Quant ML Lab", layout="wide")

st.title("Quant ML Lab")
st.caption("A growing research dashboard for applied machine learning in financial markets.")

st.markdown(
    """
This project started as a **Market Regime Detector** and is now structured as a broader
Quant ML Lab. The goal is to build separate, explainable modules for different areas of
financial machine learning: market regimes, alpha factors, volatility, options, and more.

The app is educational and research-oriented. It is not financial advice and should not be
treated as a production trading system.
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Regime Detector")
    st.markdown(
        """
Classifies market conditions such as bull trend, bear trend, high volatility,
and sideways / mean-reverting regimes using rule-based logic, KMeans, GMM, or HMM.

**Main question:** What type of market environment are we currently in?
"""
    )
    st.page_link("pages/1_Market_Regime_Detector.py", label="Open Market Regime Detector")

with col2:
    st.subheader("Alpha Factor Discovery")
    st.markdown(
        """
Tests whether candidate factors such as momentum, RSI, volatility, Bollinger z-score,
and volume z-score have a historical relationship with future returns.

**Main question:** Which signals may contain predictive information?
"""
    )
    st.page_link("pages/2_Alpha_Factor_Discovery.py", label="Open Alpha Factor Discovery")

st.divider()

st.subheader("Planned future modules")
st.markdown(
    """
- **Volatility Forecasting**: predict future volatility for risk management.
- **Options Pricing Model**: compare neural-network option pricing against Black-Scholes.
- **Portfolio Risk Dashboard**: analyze exposures, drawdowns, and risk factors.
"""
)

st.subheader("Learning focus")
st.markdown(
    """
This lab is designed to show the full workflow behind financial ML projects:

```text
market data → feature engineering → model or statistical test → visualization → validation
```

Each module is meant to be understandable, testable, and expandable.
"""
)

st.sidebar.caption("Built by San Kaneskan · Quant ML portfolio project")
