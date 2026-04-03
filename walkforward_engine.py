import pandas as pd
from strategies.strategy_registry import STRATEGIES
from experiments.grid_engine import run_grid_search_parallel, rank_results

class WalkForwardEngine:

    def __init__(self, grid_engine, backtester):

        self.grid_engine = grid_engine
        self.backtester = backtester

    def run(
        self,
        df,
        strategy_name,
        param_space,
        train_size=365 * 3,
        test_size=365,
        metric="sharpe"
    ):

        results = []

        start = 0

        while True:

            train_start = start
            train_end = start + train_size

            test_start = train_end
            test_end = test_start + test_size

            if test_end >= len(df):
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            print(
                f"\nWalk-Forward Window "
                f"Train[{train_start}:{train_end}] "
                f"Test[{test_start}:{test_end}]"
            )

            # Grid Search auf Train Set
            strategy_func = STRATEGIES[strategy_name]

            grid_results = run_grid_search_parallel(
                strategy_name,
                strategy_func,
                train_df,
                self.backtester,
                param_space
            )

            ranked = rank_results(grid_results)

            best_params = ranked[0]["params"]   

            print("Best Params:", best_params)

            # Strategie auf Testdaten laufen lassen
            strategy_func = STRATEGIES[strategy_name]

            test_strategy_df = strategy_func(test_df, **best_params)

            results_bt, _ = self.backtester.run(test_strategy_df)

            results.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "params": best_params,
                "sharpe": results_bt["sharpe"],
                "return": results_bt["total_return"],
                "max_dd": results_bt["max_drawdown"]
            })

            start += test_size

        return pd.DataFrame(results)