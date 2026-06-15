from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from market_regime_detector.alpha_factors import (
    DEFAULT_FACTOR_COLUMNS,
    build_factor_strategy,
    evaluate_factors,
    factor_strategy_metrics,
    quantile_performance,
)
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features


st.set_page_config(page_title="Alpha Factor Discovery", layout="wide")
st.title("Alpha Factor Discovery")
st.caption("Test whether today’s market features have a historical relationship with future returns.")


def section_title(text: str) -> None:
    st.subheader(text)


with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["Yahoo Finance", "Synthetic demo"], index=0)
    ticker = st.text_input("Ticker", value="SPY")
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h", "1wk", "1mo"], index=0)
    horizon = st.selectbox("Future return horizon", [1, 5, 10, 20], index=1)
    selected_factors = st.multiselect(
        "Candidate factors",
        DEFAULT_FACTOR_COLUMNS,
        default=["momentum_20", "rsi_14", "rolling_vol_20", "bb_z_20", "volume_z_20"],
        help="These are today's features tested against future returns. The future return is used only as the target, not as an input feature.",
    )
    selected_quantiles = st.slider("Quantile buckets", 3, 10, 5)
    run = st.button("Run factor discovery", type="primary")

    st.markdown("---")
    st.caption("Built by San Kaneskan · Quant ML portfolio project")

st.markdown(
    """
An **alpha factor** is a candidate signal that may contain information about future returns.
This page ranks features by their historical relationship with a selected future-return target.

The goal is not to prove a guaranteed trading edge. The goal is to explore whether some signals
show useful historical behavior worth researching further. 
"""
)

with st.expander("How alpha factor discovery works", expanded=False):
    st.markdown(
        """
1. **Load OHLCV data** from Yahoo Finance or the synthetic generator.
2. **Create features** such as momentum, RSI, volatility, Bollinger z-score, and volume z-score.
3. **Create a future return target** such as `future_return_5d`.
4. **Rank factors** using IC and Rank IC.
5. **Analyze quantiles** to see whether higher or lower factor values lead to different future returns.
6. **Backtest a simple factor rule** as an extra validation view.

The future return is used only as the target for evaluation. It is not used as an input feature.
"""
    )

if run:
    if not selected_factors:
        st.warning("Choose at least one candidate factor.")
        st.stop()

    try:
        synthetic = source == "Synthetic demo"
        chart_title = ticker if not synthetic else "Synthetic Demo"

        with st.spinner("Loading data, creating features, and ranking factors..."):
            ohlcv = load_market_data(
                MarketDataRequest(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    synthetic=synthetic,
                )
            )
            feature_df = build_features(ohlcv)
            result = evaluate_factors(feature_df, horizon=horizon, factor_columns=selected_factors)

        section_title("Factor ranking")
        st.markdown(
            """
`IC` is the correlation between today's factor value and the selected future return.
`Rank IC` is the same idea using ranked values, which can be more robust for noisy data.
"""
        )
        st.dataframe(result.ranking.style.format(precision=4), use_container_width=True)

        best_factor = result.ranking.iloc[0]["factor"]
        chosen_factor = st.selectbox(
            "Choose factor to inspect",
            result.ranking["factor"].tolist(),
            index=0,
        )

        c1, c2 = st.columns(2)
        with c1:
            section_title("Quantile performance")
            quantiles = quantile_performance(
                result.data,
                factor=chosen_factor,
                target_column=result.target_column,
                n_quantiles=selected_quantiles,
            )
            fig_q = go.Figure()
            fig_q.add_bar(
                x=quantiles["quantile"].astype(str),
                y=quantiles["avg_future_return"],
                name="Average future return",
            )
            fig_q.update_layout(
                height=420,
                title=f"{chosen_factor} vs {result.target_column}",
                xaxis_title="Factor quantile",
                yaxis_title="Average future return",
                margin=dict(l=30, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_q, use_container_width=True)
            st.dataframe(quantiles.style.format(precision=4), use_container_width=True)

        with c2:
            section_title("Factor vs future return")
            fig_s = go.Figure()
            fig_s.add_trace(
                go.Scatter(
                    x=result.data[chosen_factor],
                    y=result.data[result.target_column],
                    mode="markers",
                    name="Rows",
                    opacity=0.55,
                )
            )
            fig_s.update_layout(
                height=420,
                title=f"{chosen_factor} scatter",
                xaxis_title=chosen_factor,
                yaxis_title=result.target_column,
                margin=dict(l=30, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("A clear factor is rare in finance. Most relationships are noisy, so this chart is mainly used to inspect whether a pattern exists at all.")

        section_title("Simple factor backtest")
        strategy_df = build_factor_strategy(result.data, factor=chosen_factor, target_column=result.target_column)
        metrics = factor_strategy_metrics(strategy_df)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=strategy_df.index, y=strategy_df["benchmark_equity"], mode="lines", name="Buy & Hold"))
        fig_bt.add_trace(go.Scatter(x=strategy_df.index, y=strategy_df["factor_strategy_equity"], mode="lines", name="Factor Strategy"))
        fig_bt.update_layout(
            height=480,
            title=f"{chart_title}: simple strategy using {chosen_factor}",
            xaxis_title="Date",
            yaxis_title="Equity curve",
            hovermode="x unified",
            legend_orientation="h",
            margin=dict(l=30, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        st.dataframe(metrics.style.format(precision=4), use_container_width=True)

        section_title("Latest rows")
        display_cols = ["Close", *selected_factors, result.target_column]
        display_cols = [c for c in display_cols if c in result.data.columns]
        st.dataframe(result.data[display_cols].tail(25), use_container_width=True)

        section_title("Column guide")
        st.markdown(
            f"""
- `{result.target_column}`: the return from today's close to the close {horizon} rows in the future. This is the target used to evaluate factors.
- `IC`: Pearson correlation between the factor and the future return target.
- `Rank IC`: Spearman rank correlation between the factor and the future return target.
- `hit_rate`: how often the factor's direction agreed with the sign of the future return.
- `direction`: whether higher or lower factor values historically had the better relationship with future returns.
- `quantile`: a bucket created by sorting rows from low factor values to high factor values.
- `factor_signal`: the simple long/cash signal used in the validation backtest.

Best factor in this run by absolute Rank IC: **{best_factor}**.
"""
        )

    except Exception as exc:
        st.error(str(exc))
        st.info("Try Synthetic demo first. For Yahoo Finance, make sure yfinance is installed and you have internet access.")
else:
    st.info("Choose options in the sidebar and click **Run factor discovery**. Start with Synthetic demo + 5-day horizon if you want a no-internet test.")
