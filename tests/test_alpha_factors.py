from market_regime_detector.alpha_factors import (
    build_factor_strategy,
    evaluate_factors,
    factor_strategy_metrics,
    quantile_performance,
)
from market_regime_detector.data import MarketDataRequest, load_market_data
from market_regime_detector.features import build_features


def test_alpha_factor_discovery_smoke():
    ohlcv = load_market_data(MarketDataRequest(synthetic=True))
    features = build_features(ohlcv)
    result = evaluate_factors(features, horizon=5)
    assert not result.data.empty
    assert not result.ranking.empty
    assert result.target_column in result.data.columns

    factor = result.ranking.iloc[0]["factor"]
    quantiles = quantile_performance(result.data, factor, result.target_column)
    strategy = build_factor_strategy(result.data, factor, result.target_column)
    metrics = factor_strategy_metrics(strategy)

    assert not quantiles.empty
    assert "factor_strategy_equity" in strategy.columns
    assert not metrics.empty
