import pandas as pd


class RegimeSwitching:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # Volatility Regimes
    # --------------------------------------------------
    def volatility_regimes(self):

        df = self.df.copy()

        df["vol_regime"] = pd.qcut(df["daily_range"], 4, labels=False)

        result = df.groupby("vol_regime").agg({
            "daily_return": ["mean", "median", "std"],
            "daily_range": "mean"
        })

        result.columns = ["_".join(col) for col in result.columns]
        
        return result.to_dict()

    # --------------------------------------------------
    # Trend persistence in volatility regimes
    # --------------------------------------------------
    def trend_persistence(self):

        df = self.df.copy()

        df["next_return"] = df["daily_return"].shift(-1)
        df["vol_regime"] = pd.qcut(df["daily_range"], 4, labels=False)

        result = df.groupby("vol_regime")["next_return"].mean()

        return result

    # --------------------------------------------------
    # Direction persistence
    # --------------------------------------------------
    def direction_persistence(self):

        df = self.df.copy()

        df["next_return"] = df["daily_return"].shift(-1)

        df["same_direction"] = (
            (df["daily_return"] > 0) & (df["next_return"] > 0)
        ) | (
            (df["daily_return"] < 0) & (df["next_return"] < 0)
        )

        return df["same_direction"].mean()