import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_equity_curve(df, title="Strategy Equity Curve"):
    """
    Plot the equity curve from a dataframe that contains an 'equity' column.
    """

    if "equity" not in df.columns:
        raise ValueError("Dataframe must contain an 'equity' column")

    plt.figure(figsize=(10, 5))

    plt.plot(df["equity"], label="Strategy")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()

    plt.grid(True)

    plt.savefig("results/equity_curve.png")
    plt.close()


def plot_equity_vs_buyhold(df, title="Strategy vs Buy & Hold"):
    """
    Compare strategy equity vs buy-and-hold.
    """

    if "equity" not in df.columns:
        raise ValueError("Dataframe must contain an 'equity' column")

    if "daily_return" not in df.columns:
        raise ValueError("Dataframe must contain 'daily_return'")

    buyhold = (1 + df["daily_return"]).cumprod()

    plt.figure(figsize=(10, 5))

    plt.plot(df["equity"], label="Strategy")
    plt.plot(buyhold, label="Buy & Hold")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_parameter_heatmap(grid_results, metric="sharpe"):
    """
    Create heatmap from grid search results.
    Assumes parameters: volatility_threshold and trend_length.
    """

    rows = []

    for r in grid_results:

        params = r["params"]
        metrics = r["metrics"]

        rows.append({
            "volatility_threshold": params["volatility_threshold"],
            "trend_length": params["trend_length"],
            metric: metrics[metric]
        })

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        values=metric,
        index="volatility_threshold",
        columns="trend_length"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        pivot,
        annot=True,
        cmap="viridis",
        fmt=".2f"
    )

    plt.title(f"Parameter Heatmap ({metric})")

    plt.xlabel("Trend Length")
    plt.ylabel("Volatility Threshold")

    plt.show()
    
def plot_parameter_heatmap(grid_results, metric="sharpe"):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    rows = []

    for r in grid_results:

        params = r["params"]
        metrics = r["metrics"]

        rows.append({
            "volatility_threshold": params["volatility_threshold"],
            "trend_length": params["trend_length"],
            metric: metrics[metric]
        })

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        values=metric,
        index="volatility_threshold",
        columns="trend_length"
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        pivot,
        annot=True,
        cmap="viridis",
        fmt=".2f"
    )

    plt.title(f"Parameter Heatmap ({metric})")
    plt.xlabel("Trend Length")
    plt.ylabel("Volatility Threshold")

    plt.savefig("results/heatmap_sharpe.png")

    plt.close()

def plot_walkforward_equity(wf_results):

    """
    Plottet die Out-of-Sample Equity Curve aus Walk-Forward Ergebnissen.
    """

    equity = [1]

    for r in wf_results.to_dict("records"):

        ret = r["return"]

        new_equity = equity[-1] * (1 + ret)

        equity.append(new_equity)

    plt.figure(figsize=(10, 5))

    plt.plot(equity, linewidth=2)

    plt.title("Walk-Forward Out-of-Sample Equity Curve")

    plt.xlabel("Walk-Forward Window")

    plt.ylabel("Equity")

    plt.grid(True)

    plt.savefig("results/walkforward_equity.png")

    plt.close()

def plot_horizon_analysis(results, asset):

    plt.figure(figsize=(8,5))

    plt.plot(results.index, results["mean"], marker="o")

    plt.title(f"Horizon Analysis - {asset}")

    plt.xlabel("Holding Period (days)")
    plt.ylabel("Average Return")

    plt.grid(True)

    plt.savefig(f"results/horizon_{asset}.png")

    plt.close()

def plot_conditional_horizon(results, asset, condition):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))

    for regime in results.columns:

        plt.plot(
            results.index,
            results[regime],
            marker="o",
            label=f"{condition}={regime}"
        )

    plt.title(f"Conditional Horizon - {asset}")

    plt.xlabel("Holding Period (days)")
    plt.ylabel("Average Return")

    plt.legend()

    plt.grid(True)

    plt.savefig(f"results/conditional_horizon_{asset}.png")

    plt.close()