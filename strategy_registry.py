import numpy as np

from strategies.momentum_5d_strategy import momentum_5d_strategy
from strategies.volatility_breakout_strategy import volatility_breakout_strategy

def volatility_trend_strategy(
    df,
    volatility_threshold=75,
    trend_length=1,
    session_filter=None
):

    data = df.copy()

    # Volatility Threshold (percentile)
    threshold = df["daily_range"].quantile(volatility_threshold / 100)

    # Trend Signal (nutzt deine vorhandenen trend1–trend4 Features)
    trend_col = f"trend{trend_length}"

    if trend_col not in df.columns:
        raise ValueError(f"{trend_col} not found in dataframe")

    data["signal"] = np.where(
        df["daily_range"] > threshold,
        np.sign(df[trend_col]),
        0
    )

    return data


STRATEGIES = {
    "volatility_trend_strategy": volatility_trend_strategy,
    "momentum_5d_strategy": momentum_5d_strategy,
    "volatility_breakout_strategy": volatility_breakout_strategy
}