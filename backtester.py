import pandas as pd


class Backtester:

    def __init__(self, fee=0.0025, slippage=0.0003):

        self.fee = fee
        self.slippage = slippage

    def cost_per_trade(self):

        return self.fee + self.slippage


    def run(self, df, signal_col="signal"):

        data = df.copy()

        data["position"] = data[signal_col].shift(1).fillna(0)

        data["strategy_return"] = data["position"] * data["daily_return"]

        cost = abs(data["position"].diff()) * self.cost_per_trade()

        data["strategy_return"] -= cost

        # Trades erkennen
        data["trade"] = data["position"].diff().abs()

        num_trades = data["trade"].sum()

        # Gewinn/Verlust pro Trade approximieren
        wins = (data["strategy_return"] > 0).sum()

        total_trades = (data["strategy_return"] != 0).sum()

        if total_trades > 0:
            winrate = wins / total_trades
        else:
            winrate = 0

        # Equity Curve
        data["equity"] = (1 + data["strategy_return"]).cumprod()

        # Sharpe Ratio sicher berechnen
        std = data["strategy_return"].std()

        if std == 0 or pd.isna(std):
            sharpe = 0
        else:
            sharpe = data["strategy_return"].mean() / std * (252 ** 0.5)

        # Drawdown berechnen
        drawdown = (data["equity"] - data["equity"].cummax()) / data["equity"].cummax()

        max_dd = abs(drawdown.min())

        results = {
            "total_return": data["equity"].iloc[-1] - 1,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trades": int(num_trades),
            "winrate": float(winrate)
        }

        return results, data