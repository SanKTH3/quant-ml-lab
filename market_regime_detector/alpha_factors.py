from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


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
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    ranking: pd.DataFrame
    train_ranking: pd.DataFrame
    test_ranking: pd.DataFrame
    target_column: str
    factor_columns: list[str]
    horizon: int
    train_fraction: float
    split_date: pd.Timestamp


def add_future_return_target(feature_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add a future return target used for alpha factor testing."""
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


def _rank_factors(
    data: pd.DataFrame,
    target_column: str,
    factor_columns: list[str],
) -> pd.DataFrame:
    """Rank factors by relationship with future returns."""
    rows = []
    y = data[target_column]

    for factor in factor_columns:
        x = data[factor]

        ic = _pearson(x, y)
        rank_ic = _spearman(x, y)

        direction_value = 1.0 if pd.isna(ic) or ic >= 0 else -1.0
        centered_signal = direction_value * (x - x.median())

        predicted_positive = centered_signal > 0
        actual_positive = y > 0
        hit_rate = float((predicted_positive == actual_positive).mean())

        observations = int(pd.concat([x, y], axis=1).dropna().shape[0])

        rows.append(
            {
                "factor": factor,
                "ic": ic,
                "rank_ic": rank_ic,
                "abs_rank_ic": abs(rank_ic) if pd.notna(rank_ic) else np.nan,
                "hit_rate": hit_rate,
                "avg_future_return": float(y.mean()),
                "direction": "higher is better" if direction_value > 0 else "lower is better",
                "observations": observations,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_rank_ic", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_factors(
    feature_df: pd.DataFrame,
    horizon: int = 5,
    factor_columns: Optional[Iterable[str]] = None,
    train_fraction: float = 0.70,
) -> FactorDiscoveryResult:
    """Evaluate alpha factors with train/test validation."""
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")

    df = add_future_return_target(feature_df, horizon=horizon).sort_index()
    target = f"future_return_{horizon}d"

    factors = [c for c in (factor_columns or DEFAULT_FACTOR_COLUMNS) if c in df.columns]

    if not factors:
        raise ValueError("No valid factor columns were found in the feature dataframe.")

    df = df.dropna(subset=factors + [target]).copy()

    if len(df) < 50:
        raise ValueError("Not enough rows after feature creation. Try a longer period.")

    split_idx = int(len(df) * train_fraction)

    if split_idx < 20 or len(df) - split_idx < 20:
        raise ValueError("Not enough rows for a useful train/test split. Try a longer period.")

    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()

    train_ranking = _rank_factors(train_data, target, factors)
    test_ranking = _rank_factors(test_data, target, factors)
    full_ranking = _rank_factors(df, target, factors)

    return FactorDiscoveryResult(
        data=df,
        train_data=train_data,
        test_data=test_data,
        ranking=full_ranking,
        train_ranking=train_ranking,
        test_ranking=test_ranking,
        target_column=target,
        factor_columns=factors,
        horizon=horizon,
        train_fraction=train_fraction,
        split_date=test_data.index[0],
    )


def quantile_performance(
    data: pd.DataFrame,
    factor: str,
    target_column: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Group rows by factor value and calculate future return per bucket."""
    if factor not in data.columns:
        raise ValueError(f"Unknown factor: {factor}")

    if target_column not in data.columns:
        raise ValueError(f"Unknown target column: {target_column}")

    tmp = data[[factor, target_column]].dropna().copy()

    unique_values = tmp[factor].nunique()
    q = min(n_quantiles, unique_values)

    if q < 2:
        raise ValueError("Not enough unique factor values for quantile analysis.")

    tmp["quantile"] = pd.qcut(
        tmp[factor],
        q=q,
        labels=[f"Q{i}" for i in range(1, q + 1)],
        duplicates="drop",
    )

    return (
        tmp.groupby("quantile", observed=True)[target_column]
        .agg(
            avg_future_return="mean",
            median_future_return="median",
            count="size",
        )
        .reset_index()
    )


def get_factor_direction(ranking: pd.DataFrame, factor: str) -> str:
    """Get the factor direction from a ranking table."""
    row = ranking.loc[ranking["factor"] == factor]

    if row.empty:
        raise ValueError(f"Factor {factor} was not found in the ranking table.")

    return str(row.iloc[0]["direction"])


def factor_ranking_by_regime(
    data: pd.DataFrame,
    target_column: str,
    factor_columns: list[str],
    regime_column: str = "regime",
    min_rows: int = 20,
) -> pd.DataFrame:
    """Rank alpha factors separately inside each detected market regime."""
    if regime_column not in data.columns:
        raise ValueError(f"Regime column '{regime_column}' was not found in the dataframe.")

    rows = []

    for regime_name, regime_df in data.groupby(regime_column):
        valid_factors = [c for c in factor_columns if c in regime_df.columns]

        if len(regime_df) < min_rows or not valid_factors:
            continue

        ranking = _rank_factors(
            data=regime_df,
            target_column=target_column,
            factor_columns=valid_factors,
        )

        ranking.insert(0, "regime", regime_name)
        ranking.insert(1, "regime_rows", len(regime_df))

        rows.append(ranking)

    if not rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "regime_rows",
                "factor",
                "ic",
                "rank_ic",
                "abs_rank_ic",
                "hit_rate",
                "avg_future_return",
                "direction",
                "observations",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_factor_strategy(
    data: pd.DataFrame,
    factor: str,
    target_column: str,
    top_quantile: float = 0.80,
    direction: Optional[str] = None,
    threshold_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create a simple long/cash factor strategy."""
    if not 0.5 < top_quantile < 1.0:
        raise ValueError("top_quantile must be between 0.5 and 1.0")

    if factor not in data.columns:
        raise ValueError(f"Unknown factor: {factor}")

    df = data.copy()
    source = threshold_data if threshold_data is not None else df

    if direction is None:
        ic = _pearson(source[factor], source[target_column])
        direction_value = 1.0 if pd.isna(ic) or ic >= 0 else -1.0
    else:
        direction_value = 1.0 if direction == "higher is better" else -1.0

    if direction_value > 0:
        threshold = source[factor].quantile(top_quantile)
        raw_signal = (df[factor] >= threshold).astype(float)
    else:
        threshold = source[factor].quantile(1 - top_quantile)
        raw_signal = (df[factor] <= threshold).astype(float)

    df["asset_return"] = df["Close"].pct_change().fillna(0)
    df["factor_signal"] = raw_signal.shift(1).fillna(0)
    df["factor_strategy_return"] = df["factor_signal"] * df["asset_return"]

    df["benchmark_equity"] = (1 + df["asset_return"]).cumprod()
    df["factor_strategy_equity"] = (1 + df["factor_strategy_return"]).cumprod()

    return df


def factor_strategy_metrics(
    strategy_df: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate basic metrics for buy-and-hold and the factor strategy."""
    rows = []

    for name, col in [
        ("Buy & Hold", "asset_return"),
        ("Factor Strategy", "factor_strategy_return"),
    ]:
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