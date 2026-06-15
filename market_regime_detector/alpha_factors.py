from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURE_COLUMNS

DEFAULT_FACTOR_COLUMNS = [
    "momentum_10",
    "momentum_20",
    "ma_slope_20",
    "ma_slope_50",
    "rsi_14",
    "rolling_vol_20",
    "rolling_vol_60",
    "atr_pct_14",
    "autocorr_20",
    "bb_z_20",
    "volume_z_20",
]


@dataclass
class FactorDiscoveryResult:
    data: pd.DataFrame
    ranking: pd.DataFrame
    target_column: str
    factor_columns: list[str]
    horizon: int


def add_future_return_target(feature_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add a future return target used for alpha factor testing.

    The target is the return from today's close to the close `horizon` rows in
    the future. It is used only for evaluation, not as an input feature.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    df = feature_df.copy()
    target = f"future_return_{horizon}d"
    df[target] = df["Close"].pct_change(horizon).shift(-horizon)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if df.empty:
        raise ValueError("Not enough rows to create the selected future return target.")
    return df


def _pearson(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def evaluate_factors(
    feature_df: pd.DataFrame,
    horizon: int = 5,
    factor_columns: Optional[Iterable[str]] = None,
) -> FactorDiscoveryResult:
    """Rank candidate factors by historical relationship with future returns."""
    df = add_future_return_target(feature_df, horizon=horizon)
    target = f"future_return_{horizon}d"
    factors = [c for c in (factor_columns or DEFAULT_FACTOR_COLUMNS) if c in df.columns]
    if not factors:
        raise ValueError("No valid factor columns were found in the feature dataframe.")

    rows = []
    y = df[target]
    for factor in factors:
        x = df[factor]
        ic = _pearson(x, y)
        rank_ic = _spearman(x, y)
        direction = 1.0 if pd.isna(ic) or ic >= 0 else -1.0

        centered_signal = direction * (x - x.median())
        predicted_positive = centered_signal > 0
        actual_positive = y > 0
        hit_rate = float((predicted_positive == actual_positive).mean())

        rows.append(
            {
                "factor": factor,
                "ic": ic,
                "rank_ic": rank_ic,
                "abs_rank_ic": abs(rank_ic) if pd.notna(rank_ic) else np.nan,
                "hit_rate": hit_rate,
                "avg_future_return": float(y.mean()),
                "direction": "higher is better" if direction > 0 else "lower is better",
            }
        )

    ranking = pd.DataFrame(rows).sort_values("abs_rank_ic", ascending=False).reset_index(drop=True)
    return FactorDiscoveryResult(data=df, ranking=ranking, target_column=target, factor_columns=factors, horizon=horizon)


def quantile_performance(
    data: pd.DataFrame,
    factor: str,
    target_column: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Group dates by factor value and calculate average future return per bucket."""
    if factor not in data.columns:
        raise ValueError(f"Unknown factor: {factor}")
    if target_column not in data.columns:
        raise ValueError(f"Unknown target column: {target_column}")

    tmp = data[[factor, target_column]].dropna().copy()
    unique_values = tmp[factor].nunique()
    q = min(n_quantiles, unique_values)
    if q < 2:
        raise ValueError("Not enough unique factor values for quantile analysis.")

    tmp["quantile"] = pd.qcut(tmp[factor], q=q, labels=[f"Q{i}" for i in range(1, q + 1)], duplicates="drop")
    out = (
        tmp.groupby("quantile", observed=True)[target_column]
        .agg(avg_future_return="mean", median_future_return="median", count="size")
        .reset_index()
    )
    return out


def build_factor_strategy(
    data: pd.DataFrame,
    factor: str,
    target_column: str,
    top_quantile: float = 0.80,
) -> pd.DataFrame:
    """Create a simple long/cash strategy from one selected factor.

    If the factor's IC is positive, the strategy goes long after high factor
    values. If the IC is negative, it goes long after low factor values. The
    signal is shifted by one row so today's signal is used on the next row's
    return, reducing basic look-ahead bias.
    """
    if not 0.5 < top_quantile < 1.0:
        raise ValueError("top_quantile must be between 0.5 and 1.0")

    df = data.copy()
    ic = _pearson(df[factor], df[target_column])
    direction = 1.0 if pd.isna(ic) or ic >= 0 else -1.0

    if direction > 0:
        threshold = df[factor].quantile(top_quantile)
        raw_signal = (df[factor] >= threshold).astype(float)
    else:
        threshold = df[factor].quantile(1 - top_quantile)
        raw_signal = (df[factor] <= threshold).astype(float)

    df["asset_return"] = df["Close"].pct_change().fillna(0)
    df["factor_signal"] = raw_signal.shift(1).fillna(0)
    df["factor_strategy_return"] = df["factor_signal"] * df["asset_return"]
    df["benchmark_equity"] = (1 + df["asset_return"]).cumprod()
    df["factor_strategy_equity"] = (1 + df["factor_strategy_return"]).cumprod()
    return df


def factor_strategy_metrics(strategy_df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Calculate basic metrics for buy-and-hold and the factor strategy."""
    rows = []
    for name, col in [("Buy & Hold", "asset_return"), ("Factor Strategy", "factor_strategy_return")]:
        r = strategy_df[col].dropna()
        if r.empty:
            continue
        total_return = (1 + r).prod() - 1
        years = max(len(r) / periods_per_year, 1 / periods_per_year)
        cagr = (1 + total_return) ** (1 / years) - 1
        vol = r.std() * np.sqrt(periods_per_year)
        sharpe = np.nan if vol == 0 else (r.mean() * periods_per_year) / vol
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "strategy": name,
                "total_return": total_return,
                "cagr": cagr,
                "annualized_vol": vol,
                "sharpe_no_rf": sharpe,
                "max_drawdown": drawdown.min(),
            }
        )
    return pd.DataFrame(rows)
