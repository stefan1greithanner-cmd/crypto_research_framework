import pandas as pd


class DescriptiveStats:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --------------------------------------------------
    # Basic Market Statistics
    # --------------------------------------------------
    def basic_stats(self):

        stats = {
            "mean_daily_return": self.df["daily_return"].mean(),
            "median_daily_return": self.df["daily_return"].median(),
            "std_daily_return": self.df["daily_return"].std(),

            "mean_daily_range": self.df["daily_range"].mean(),
            "median_daily_range": self.df["daily_range"].median(),
            "std_daily_range": self.df["daily_range"].std(),
        }

        return pd.Series(stats)

    # --------------------------------------------------
    # Weekday Analysis
    # --------------------------------------------------
    def weekday_returns(self):

        return self.df.groupby("weekday")["daily_return"].agg(
            ["mean", "median", "std", "count"]
        )

    # --------------------------------------------------
    # Weekend vs Weekday
    # --------------------------------------------------
    def weekend_effect(self):

        return self.df.groupby("is_weekend")["daily_return"].agg(
            ["mean", "median", "std", "count"]
        )

    # --------------------------------------------------
    # Trend Conditional Returns
    # --------------------------------------------------
    def trend_conditional(self, threshold=0.03):

        trend_up = self.df[self.df["trend3"] > threshold]
        trend_down = self.df[self.df["trend3"] < -threshold]

        result = pd.DataFrame({
            "condition": ["trend_up", "trend_down"],
            "mean_return": [
                trend_up["daily_return"].mean(),
                trend_down["daily_return"].mean()
            ],
            "count": [
                len(trend_up),
                len(trend_down)
            ]
        })

        return result

    # --------------------------------------------------
    # Session Volatility
    # --------------------------------------------------
    def session_volatility(self):

        cols = [
            "range_00_06",
            "range_06_12",
            "range_12_18",
            "range_18_24"
        ]

        return self.df[cols].mean().sort_values(ascending=False)