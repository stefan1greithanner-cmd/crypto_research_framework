import pandas as pd


def volatility_breakout_strategy(
    df,
    lookback=10,
    compression_quantile=0.2,
    holding=10
):

    data = df.copy()

    # Rolling Range
    data["range"] = data["high"] - data["low"]

    data["range_mean"] = data["range"].rolling(lookback).mean()

    # Volatility Compression Threshold
    threshold = data["range_mean"].quantile(compression_quantile)

    data["compression"] = data["range_mean"] < threshold

    # Breakout Level
    data["highest_high"] = data["high"].rolling(lookback).max()

    # Breakout Signal
    data["signal"] = 0

    breakout = (data["close"] > data["highest_high"].shift(1)) & data["compression"]

    data.loc[breakout, "signal"] = 1

    # Holding period
    position = data["signal"].copy()

    for i in range(1, holding):

        position = position.combine_first(
            data["signal"].shift(i)
        )

    data["signal"] = position.fillna(0)

    return data