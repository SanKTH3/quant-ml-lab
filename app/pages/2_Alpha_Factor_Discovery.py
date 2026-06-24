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
    factor_ranking_by_regime,
    factor_strategy_metrics,
    get_factor_direction,
    quantile_performance,
)
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features
from market_regime_detector.regimes import RegimeDetector


st.set_page_config(page_title="Alpha Factor Discovery", layout="wide")

st.title("Alpha Factor Discovery")
st.caption("Test whether today’s market signals have a historical relationship with future returns.")


def section_title(text: str) -> None:
    st.subheader(text)


def fmt_number(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_percent(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def get_factor_row(ranking: pd.DataFrame, factor: str) -> pd.Series | None:
    rows = ranking.loc[ranking["factor"] == factor]
    if rows.empty:
        return None
    return rows.iloc[0]


def sign_label(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def interpret_train_test(train_rank_ic: float, test_rank_ic: float, factor: str) -> str:
    train_sign = sign_label(train_rank_ic)
    test_sign = sign_label(test_rank_ic)

    if train_sign == "unknown" or test_sign == "unknown":
        return (
            f"The app could not calculate a clear Rank IC for `{factor}`. "
            "This usually means there were too few valid rows or too much missing data."
        )

    if train_sign == "flat" or test_sign == "flat":
        return (
            f"`{factor}` has a weak relationship with future returns in at least one period. "
            "This suggests the factor is not very strong for the selected settings."
        )

    if train_sign == test_sign:
        return (
            f"`{factor}` kept the same direction in both the training and test periods. "
            "That is a better sign because the relationship did not immediately reverse out-of-sample."
        )

    return (
        f"`{factor}` changed direction between the training and test periods. "
        "That suggests the factor was unstable and may depend on the market environment."
    )


def explain_direction(direction: str) -> str:
    if direction == "higher is better":
        return "Higher factor values were better during the training period."
    return "Lower factor values were better during the training period."


with st.sidebar:
    st.header("Configuration")

    beginner_mode = st.checkbox(
        "Beginner mode",
        value=True,
        help="Shows only the main conclusions first and hides advanced tables inside expanders.",
    )

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
        help=(
            "The first part of the data is used for factor research. "
            "The later part is used for out-of-sample validation."
        ),
    )

    regime_method = st.selectbox(
        "Regime model for factor analysis",
        ["gmm", "kmeans", "rule", "hmm"],
        index=0,
        help=(
            "Used to detect market regimes before checking how alpha factors "
            "behave inside each regime."
        ),
    )

    n_regimes = st.slider(
        "Number of regimes",
        min_value=2,
        max_value=6,
        value=4,
        help="Used for KMeans, GMM, and HMM. Rule-based mode uses its own rules.",
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


if beginner_mode:
    st.info(
        "Beginner mode is on. This page will first show the main conclusion, "
        "then hide the detailed research tables under Advanced Details."
    )


st.markdown(
    """
This page tries to answer one simple question:

**Did a market signal look useful in the past, and did it still work later?**

The app does this by testing factors such as momentum, RSI, volatility, and volume against future returns.
"""
)


with st.expander("How this page works", expanded=False):
    st.markdown(
        """
1. Load market data.
2. Create financial features such as momentum, RSI, volatility, and volume z-score.
3. Detect market regimes such as bull, bear, sideways, or volatile markets.
4. Create a future return target, for example `future_return_5d`.
5. Use the first part of the data as the training period.
6. Use the later part as the test period.
7. Check whether the selected factor still behaves well out-of-sample.
8. Hide advanced tables unless you want to inspect them.
"""
    )


current_config = {
    "source": source,
    "ticker": ticker,
    "period": period,
    "interval": interval,
    "horizon": horizon,
    "train_fraction": train_fraction,
    "regime_method": regime_method,
    "n_regimes": n_regimes,
    "selected_factors": tuple(selected_factors),
}


if run:
    if not selected_factors:
        st.warning("Choose at least one candidate factor.")
        st.stop()

    try:
        synthetic = source == "Synthetic demo"
        chart_title = ticker if not synthetic else "Synthetic Demo"

        with st.spinner("Loading data, detecting regimes, and testing factors..."):
            ohlcv = load_market_data(
                MarketDataRequest(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    synthetic=synthetic,
                )
            )

            feature_df = build_features(ohlcv)

            regime_detector = RegimeDetector(
                method=regime_method,
                n_regimes=n_regimes,
            )

            regime_result = regime_detector.fit_predict(feature_df)

            result = evaluate_factors(
                regime_result.data,
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


if "alpha_analysis" not in st.session_state:
    st.info(
        "Choose options in the sidebar and click **Run factor discovery**. "
        "Start with Synthetic demo + 5-day horizon if you want a simple test."
    )
    st.stop()


analysis = st.session_state["alpha_analysis"]

result = analysis["result"]
chart_title = analysis["chart_title"]
saved_config = analysis["config"]

if saved_config != current_config:
    st.warning(
        "The sidebar settings have changed. Click **Run factor discovery** again "
        "to recompute the analysis with the new settings."
    )

best_factor = result.train_ranking.iloc[0]["factor"]

chosen_factor = st.selectbox(
    "Choose factor to inspect",
    result.train_ranking["factor"].tolist(),
    index=0,
    help="The list is ordered by in-sample Rank IC.",
)

chosen_direction = get_factor_direction(result.train_ranking, chosen_factor)

train_row = get_factor_row(result.train_ranking, chosen_factor)
test_row = get_factor_row(result.test_ranking, chosen_factor)

train_rank_ic = train_row["rank_ic"] if train_row is not None else float("nan")
test_rank_ic = test_row["rank_ic"] if test_row is not None else float("nan")
train_ic = train_row["ic"] if train_row is not None else float("nan")
test_ic = test_row["ic"] if test_row is not None else float("nan")
train_hit_rate = train_row["hit_rate"] if train_row is not None else float("nan")
test_hit_rate = test_row["hit_rate"] if test_row is not None else float("nan")

st.success(
    f"Analysis complete. Training period: first {int(result.train_fraction * 100)}% "
    f"of rows. Test period starts at: {result.split_date.date()}."
)


# ------------------------------------------------------------
# 1. Simple Summary
# ------------------------------------------------------------

section_title("1. Simple Summary")

summary_cols = st.columns(4)

with summary_cols[0]:
    st.metric("Selected factor", chosen_factor)

with summary_cols[1]:
    st.metric("Training Rank IC", fmt_number(train_rank_ic))

with summary_cols[2]:
    st.metric("Test Rank IC", fmt_number(test_rank_ic))

with summary_cols[3]:
    st.metric("Test hit rate", fmt_percent(test_hit_rate))


st.markdown(
    f"""
**Plain-English interpretation**

{interpret_train_test(train_rank_ic, test_rank_ic, chosen_factor)}

**Direction learned from training period:** `{chosen_direction}`  
{explain_direction(chosen_direction)}
"""
)


# ------------------------------------------------------------
# 2. Main Factor Result
# ------------------------------------------------------------

section_title("2. Main Factor Result")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Training period")
    st.write(
        f"""
- IC: `{fmt_number(train_ic)}`
- Rank IC: `{fmt_number(train_rank_ic)}`
- Hit rate: `{fmt_percent(train_hit_rate)}`
- Direction: `{chosen_direction}`
"""
    )

with c2:
    st.markdown("#### Test period")
    st.write(
        f"""
- IC: `{fmt_number(test_ic)}`
- Rank IC: `{fmt_number(test_rank_ic)}`
- Hit rate: `{fmt_percent(test_hit_rate)}`
"""
    )


# ------------------------------------------------------------
# 3. Out-of-Sample Test
# ------------------------------------------------------------

section_title("3. Out-of-Sample Test")

st.markdown(
    """
This chart checks the selected factor only on the **test period**.

The factor values are split into buckets from low values to high values.
A clearer pattern means the factor may have more useful information.
"""
)

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
        xaxis_title="Factor bucket",
        yaxis_title="Average future return",
        margin=dict(l=30, r=20, t=60, b=40),
    )

    st.plotly_chart(fig_q, use_container_width=True)

    if not beginner_mode:
        st.dataframe(
            quantiles_test.style.format(precision=4),
            use_container_width=True,
        )

except Exception as exc:
    st.warning(f"Could not create quantile chart: {exc}")


# ------------------------------------------------------------
# 4. Simple Backtest
# ------------------------------------------------------------

section_title("4. Simple Backtest")

st.markdown(
    """
This backtest learns the factor rule from the **training period**, then applies it to the full dataset.

The vertical dashed line shows when the test period starts.  
The most important part is what happens **after** that line.
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
    line_color="red",
    line_width=3,
    annotation_text="Test starts",
    annotation_position="top",
    annotation_font_color="red",
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

st.markdown("#### Backtest metrics")
st.dataframe(metrics.style.format(precision=4), use_container_width=True)


# ------------------------------------------------------------
# 5. Advanced Details
# ------------------------------------------------------------

section_title("5. Advanced Details")

with st.expander("Open advanced research tables and charts", expanded=not beginner_mode):
    st.markdown(
        """
These sections are useful when you want to inspect the full research output.
You do not need to understand every table immediately.
"""
    )

    st.markdown("### Full factor ranking tables")

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

    st.markdown("### Alpha factors by market regime")

    st.markdown(
        """
This table checks whether factors behave differently inside different detected market regimes.
For example, momentum may behave differently in bull markets than in high-volatility markets.
"""
    )

    try:
        regime_factor_table = factor_ranking_by_regime(
            data=result.test_data,
            target_column=result.target_column,
            factor_columns=result.factor_columns,
            regime_column="regime",
        )

        if regime_factor_table.empty:
            st.warning("Not enough rows to calculate regime-specific factor rankings.")
        else:
            st.dataframe(
                regime_factor_table.style.format(precision=4),
                use_container_width=True,
            )

    except Exception as exc:
        st.warning(f"Could not calculate regime-specific factor rankings: {exc}")

    st.markdown("### Scatter plot")

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

    st.markdown("### In-sample quantile performance")

    try:
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
            xaxis_title="Factor bucket",
            yaxis_title="Average future return",
            margin=dict(l=30, r=20, t=60, b=40),
        )

        st.plotly_chart(fig_q_train, use_container_width=True)
        st.dataframe(
            quantiles_train.style.format(precision=4),
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(f"Could not create in-sample quantile chart: {exc}")

    st.markdown("### Latest rows")

    display_cols = [
        "Close",
        "regime",
        *saved_config["selected_factors"],
        result.target_column,
    ]
    display_cols = [c for c in display_cols if c in result.data.columns]

    st.dataframe(
        result.data[display_cols].tail(25),
        use_container_width=True,
    )

    st.markdown("### Column guide")

    st.markdown(
        f"""
- `{result.target_column}`: return from today's close to the close {saved_config["horizon"]} rows in the future.
- `IC`: Pearson correlation between the factor and the future return target.
- `Rank IC`: Spearman rank correlation between the factor and the future return target.
- `hit_rate`: how often the factor's direction agreed with the sign of the future return.
- `direction`: whether higher or lower factor values historically had the better relationship with future returns.
- `quantile`: a bucket created by sorting rows from low factor values to high factor values.
- `factor_signal`: the simple long/cash signal used in the validation backtest.
- `regime`: the detected market environment from the selected regime model.
- `In-sample`: the earlier part of the data used to research and select factors.
- `Out-of-sample`: the later part of the data used to validate whether the factor still works.

Best factor in this run by in-sample absolute Rank IC: **{best_factor}**.
"""
    )