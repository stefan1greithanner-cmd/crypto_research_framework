import pandas as pd


def horizon_analysis(df, horizons=[1,2,3,5,7,10,14,21,30]):

    """
    Analyse der durchschnittlichen Returns über verschiedene Holding Periods.
    """

    results = {}

    for h in horizons:

        future_return = df["close"].pct_change(h).shift(-h)

        results[h] = {
            "mean": future_return.mean(),
            "median": future_return.median(),
            "std": future_return.std()
        }

    return pd.DataFrame(results).T