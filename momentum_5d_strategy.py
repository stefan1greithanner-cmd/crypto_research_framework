import pandas as pd


def momentum_5d_strategy(
    df,
    lookback=5,
    holding=10,
    threshold=0.02
):

    data = df.copy()

    # Momentum Signal
    data["momentum"] = data["close"].pct_change(lookback)

    data["signal"] = 0

    data.loc[data["momentum"] > threshold, "signal"] = 1

    # Holding Period erweitern
    position = data["signal"].copy()

    for i in range(1, holding):

        position = position.combine_first(
            data["signal"].shift(i)
        )

    data["signal"] = position.fillna(0)

    return data