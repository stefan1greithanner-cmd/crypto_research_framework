import pandas as pd


def conditional_horizon_analysis(
    df,
    condition_col,
    horizons=[1,2,3,5,7,10,14,21,30]
):

    results = {}

    condition_values = df[condition_col].dropna().unique()

    for cond in sorted(condition_values):

        subset = df[df[condition_col] == cond]

        horizon_stats = {}

        for h in horizons:

            future_return = subset["close"].pct_change(h).shift(-h)

            horizon_stats[h] = future_return.mean()

        results[cond] = horizon_stats

    return pd.DataFrame(results)