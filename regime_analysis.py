import pandas as pd


class RegimeAnalysis:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # Quiet Morning → US Session Expansion
    # --------------------------------------------------
    def quiet_morning_expansion(self):

        df = self.df.copy()

        df["morning_range"] = df["range_00_06"] + df["range_06_12"]
        df["us_range"] = df["range_12_18"]

        # Quartile der Morgenvolatilität
        df["morning_quartile"] = pd.qcut(df["morning_range"], 4, labels=False)

        result = df.groupby("morning_quartile")["us_range"].agg(
            ["mean", "median", "std", "count"]
        )

        return result

    # --------------------------------------------------
    # Volatility Clustering
    # --------------------------------------------------
    def volatility_clustering(self):

        df = self.df.copy()

        df["prev_range"] = df["daily_range"].shift(1)

        df["range_quartile"] = pd.qcut(df["prev_range"], 4, labels=False)

        result = df.groupby("range_quartile")["daily_range"].agg(
            ["mean", "median", "std", "count"]
        )

        return result

    # --------------------------------------------------
    # Big Move Follow Through
    # --------------------------------------------------
    def big_move_follow_through(self, threshold=0.06):

        df = self.df.copy()
        
        df["next_return"] = df["daily_return"].shift(-1)

        big_up = df[df["daily_return"] > threshold]
        big_down = df[df["daily_return"] < -threshold]

        result = pd.DataFrame({
            "condition": ["big_up", "big_down"],
            "next_day_return_mean": [
                big_up["daily_return"].shift(-1).mean(),
                big_down["daily_return"].shift(-1).mean()
            ],
            "count": [
                len(big_up),
                len(big_down)
            ]
        })

        return result