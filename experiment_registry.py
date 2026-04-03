from analysis.descriptive_stats import DescriptiveStats
from analysis.regime_analysis import RegimeAnalysis
from analysis.regime_switching import RegimeSwitching


def descriptive_experiment(df):

    stats = DescriptiveStats(df)

    return {
        "basic_stats": stats.basic_stats().to_dict(),
        "weekday_returns": stats.weekday_returns().to_dict()
    }


def regime_experiment(df):

    regime = RegimeAnalysis(df)

    return {
        "quiet_morning": regime.quiet_morning_expansion().to_dict(),
        "vol_clustering": regime.volatility_clustering().to_dict()
    }


def regime_switch_experiment(df):

    regimes = RegimeSwitching(df)

    return {
        "volatility_regimes": regimes.volatility_regimes(),
        "trend_persistence": regimes.trend_persistence().to_dict(),
        "direction_persistence": regimes.direction_persistence()
    }


EXPERIMENTS = {
    "descriptive": descriptive_experiment,
    "regime": regime_experiment,
    "regime_switch": regime_switch_experiment
}