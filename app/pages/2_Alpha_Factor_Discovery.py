from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_regime_detector.alpha_factors import (
    DEFAULT_FACTOR_COLUMNS,
    build_factor_strategy,
    evaluate_factors,
    factor_strategy_metrics,
    get_factor_direction,
    quantile_performance,
)
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features


st.set_page_config(page_title="Alpha Factor Discovery", layout="wide")

st.title("Alpha Factor Discovery")
st.caption(
    "Test whether today’s market features have a historical relationship with future returns."
)


def section_title(text: str) -> None:
    st.subheader(text)


with st.sidebar:
    st.header("Configuration")

    source = st.radio("Data source", ["Yahoo Finance", "Synthetic demo"], index=0)
    ticker = st.text_input("Ticker", value="SPY")
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h", "1wk", "1mo"], index=0)
    horizon = st.selectbox("Future return horizon", [1, 5, 10, 20], index=1)

    train_fraction = st.slider(
        "Training period size",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="The first part of the data is used for factor research. The later part is used for out-of-sample validation.",
    )

    selected_factors = st.multiselect(
        "Candidate factors",
        DEFAULT_FACTOR_COLUMNS,
        default=[
            "momentum_20",
            "rsi_14",
            "rolling_vol_20",
            "bb_z_20",
            "volume_z_20",
        ],
        help=(
            "These are today's features tested against future returns. "
            "The future return is used only as the target, not as an input feature."
        ),
    )

    selected_quantiles = st.slider("Quantile buckets", 3, 10, 5)

    run = st.button("Run factor discovery", type="primary")

    st.markdown("---")
    st.caption("Built by San Kaneskan · Quant ML portfolio project")


st.markdown(
    """
An **alpha factor** is a candidate signal that may contain information about future returns.

This page now separates the analysis into:

- **In-sample period**: used to research and rank factors
- **Out-of-sample period**: used to check whether the factor still behaves well on unseen data

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
4. **Split the data** into an in-sample period and out-of-sample period.
5. **Rank factors** on the in-sample period using IC and Rank IC.
6. **Check out-of-sample results** to see whether the relationship still appears later.
7. **Backtest a simple factor rule** as an extra validation view.

The future return is used only as the target for evaluation. It is not used as an input feature.
"""
    )


current_config = {
    "source": source,
    "ticker": ticker,
    "period": period,
    "interval": interval,
    "horizon": horizon,
    "train_fraction": train_fraction,
    "selected_factors": tuple(selected_factors),
}


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

            result = evaluate_factors(
                feature_df,
                horizon=horizon,
                factor_columns=selected_factors,
                train_fraction=train_fraction,
            )

        st.session_state["alpha_analysis"] = {
            "result": result,
            "chart_title": chart_title,
            "config": current_config,
        }

    except Exception as exc:
        st.error(str(exc))
        st.info(
            "Try Synthetic demo first. For Yahoo Finance, make sure yfinance is installed "
            "and you have internet access."
        )


if "alpha_analysis" in st.session_state:
    analysis = st.session_state["alpha_analysis"]

    result = analysis["result"]
    chart_title = analysis["chart_title"]
    saved_config = analysis["config"]

    if saved_config != current_config:
        st.warning(
            "The sidebar settings have changed. Click **Run factor discovery** again "
            "to recompute the analysis with the new settings."
        )

    st.success(
        f"Analysis complete. Training period: first {int(result.train_fraction * 100)}% "
        f"of rows. Test period starts at: {result.split_date.date()}."
    )

    section_title("Factor ranking")

    st.markdown(
        """
`IC` is the correlation between today's factor value and the selected future return.
`Rank IC` is the same idea using ranked values, which can be more robust for noisy data.

The **in-sample ranking** is used to discover promising factors.
The **out-of-sample ranking** checks whether those relationships still appear later.
"""
    )

    tab_train, tab_test, tab_full = st.tabs(
        ["In-sample ranking", "Out-of-sample ranking", "Full-sample ranking"]
    )

    with tab_train:
        st.dataframe(
            result.train_ranking.style.format(precision=4),
            use_container_width=True,
        )

    with tab_test:
        st.dataframe(
            result.test_ranking.style.format(precision=4),
            use_container_width=True,
        )

    with tab_full:
        st.dataframe(
            result.ranking.style.format(precision=4),
            use_container_width=True,
        )

    best_factor = result.train_ranking.iloc[0]["factor"]

    chosen_factor = st.selectbox(
        "Choose factor to inspect",
        result.train_ranking["factor"].tolist(),
        index=0,
        help="The list is ordered by in-sample Rank IC. This simulates choosing a factor during the research period.",
    )

    chosen_direction = get_factor_direction(result.train_ranking, chosen_factor)

    st.info(
        f"Selected factor: **{chosen_factor}**. "
        f"Direction learned from training period: **{chosen_direction}**."
    )

    c1, c2 = st.columns(2)

    with c1:
        section_title("Out-of-sample quantile performance")

        try:
            quantiles_test = quantile_performance(
                result.test_data,
                factor=chosen_factor,
                target_column=result.target_column,
                n_quantiles=selected_quantiles,
            )

            fig_q = go.Figure()
            fig_q.add_bar(
                x=quantiles_test["quantile"].astype(str),
                y=quantiles_test["avg_future_return"],
                name="Average future return",
            )
            fig_q.update_layout(
                height=420,
                title=f"Test period: {chosen_factor} vs {result.target_column}",
                xaxis_title="Factor quantile",
                yaxis_title="Average future return",
                margin=dict(l=30, r=20, t=60, b=40),
            )

            st.plotly_chart(fig_q, use_container_width=True)
            st.dataframe(
                quantiles_test.style.format(precision=4),
                use_container_width=True,
            )

        except Exception as exc:
            st.warning(f"Could not create quantile chart: {exc}")

    with c2:
        section_title("Out-of-sample factor vs future return")

        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=result.test_data[chosen_factor],
                y=result.test_data[result.target_column],
                mode="markers",
                name="Test rows",
                opacity=0.55,
            )
        )
        fig_s.update_layout(
            height=420,
            title=f"Test period scatter: {chosen_factor}",
            xaxis_title=chosen_factor,
            yaxis_title=result.target_column,
            margin=dict(l=30, r=20, t=60, b=40),
        )

        st.plotly_chart(fig_s, use_container_width=True)
        st.caption(
            "A clear factor is rare in finance. Most relationships are noisy, "
            "so this chart is mainly used to inspect whether a pattern exists at all."
        )

    with st.expander("Show in-sample quantile performance", expanded=False):
        quantiles_train = quantile_performance(
            result.train_data,
            factor=chosen_factor,
            target_column=result.target_column,
            n_quantiles=selected_quantiles,
        )

        fig_q_train = go.Figure()
        fig_q_train.add_bar(
            x=quantiles_train["quantile"].astype(str),
            y=quantiles_train["avg_future_return"],
            name="Average future return",
        )
        fig_q_train.update_layout(
            height=420,
            title=f"Training period: {chosen_factor} vs {result.target_column}",
            xaxis_title="Factor quantile",
            yaxis_title="Average future return",
            margin=dict(l=30, r=20, t=60, b=40),
        )

        st.plotly_chart(fig_q_train, use_container_width=True)
        st.dataframe(
            quantiles_train.style.format(precision=4),
            use_container_width=True,
        )

    section_title("Simple factor backtest")

    st.markdown(
        """
This backtest learns the factor direction and threshold from the **training period**,
then applies the same rule across the full dataset.

This is more realistic than choosing the rule using all historical data.
"""
    )

    strategy_df = build_factor_strategy(
        result.data,
        factor=chosen_factor,
        target_column=result.target_column,
        direction=chosen_direction,
        threshold_data=result.train_data,
    )

    train_strategy_df = strategy_df.loc[strategy_df.index < result.split_date]
    test_strategy_df = strategy_df.loc[strategy_df.index >= result.split_date]

    train_metrics = factor_strategy_metrics(train_strategy_df)
    train_metrics.insert(0, "period", "In-sample")

    test_metrics = factor_strategy_metrics(test_strategy_df)
    test_metrics.insert(0, "period", "Out-of-sample")

    metrics = pd.concat([train_metrics, test_metrics], ignore_index=True)

    fig_bt = go.Figure()

    fig_bt.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["benchmark_equity"],
            mode="lines",
            name="Buy & Hold",
        )
    )

    fig_bt.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["factor_strategy_equity"],
            mode="lines",
            name="Factor Strategy",
        )
    )

    fig_bt.add_vline(
        x=result.split_date,
        line_dash="dash",
        annotation_text="Test starts",
        annotation_position="top",
    )

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

    display_cols = [
        "Close",
        *saved_config["selected_factors"],
        result.target_column,
    ]
    display_cols = [c for c in display_cols if c in result.data.columns]

    st.dataframe(
        result.data[display_cols].tail(25),
        use_container_width=True,
    )

    section_title("Column guide")

    st.markdown(
        f"""
- `{result.target_column}`: the return from today's close to the close {saved_config["horizon"]} rows in the future.
- `IC`: Pearson correlation between the factor and the future return target.
- `Rank IC`: Spearman rank correlation between the factor and the future return target.
- `hit_rate`: how often the factor's direction agreed with the sign of the future return.
- `direction`: whether higher or lower factor values historically had the better relationship with future returns.
- `quantile`: a bucket created by sorting rows from low factor values to high factor values.
- `factor_signal`: the simple long/cash signal used in the validation backtest.
- `In-sample`: the earlier part of the data used to research and select factors.
- `Out-of-sample`: the later part of the data used to validate whether the factor still works.

Best factor in this run by in-sample absolute Rank IC: **{best_factor}**.
"""
    )

else:
    st.info(
        "Choose options in the sidebar and click **Run factor discovery**. "
        "Start with Synthetic demo + 5-day horizon if you want a no-internet test."
    )