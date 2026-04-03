import json
import os
from datetime import datetime
import pandas as pd

from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from experiments.experiment_registry import EXPERIMENTS

from core.backtester import Backtester
from strategies.strategy_registry import STRATEGIES

from strategies.strategy_params import PARAMETER_SPACE
from analysis.research_plots import plot_equity_curve

from analysis.research_plots import plot_parameter_heatmap
from experiments.grid_engine import (
    run_grid_search_parallel,
    rank_results
)
from experiments.walkforward_engine import WalkForwardEngine
from analysis.research_plots import plot_walkforward_equity

from analysis.horizon_analysis import horizon_analysis
from analysis.research_plots import plot_horizon_analysis

from analysis.conditional_horizon_analysis import conditional_horizon_analysis
from analysis.research_plots import plot_conditional_horizon

class ResearchAgent:

    def __init__(self, data_path):

        self.loader = DataLoader(data_path)
        self.engine = FeatureEngine()
        self.backtester = Backtester()

    def prepare_data(self):

        asset_files = {
            "BTC": "btc_1m_all.parquet",
            "ETH": "eth_1m_all.parquet",
            "LINK": "link_1m_all.parquet"
        }

        assets = self.loader.load_multiple_assets(asset_files)

        features = {}

        for asset, df in assets.items():

            print(f"Preparing data for {asset}")

            df = self.loader.filter_date_range(df, start="2022-01-01")

            df_1h = self.loader.resample_ohlcv(df, "1h")

            features[asset] = self.engine.build_daily_structure(df_1h)

        return features

    def run_backtests(self, data):

        backtest_results = {}

        for asset, df in data.items():

            print(f"\n===== BACKTESTING {asset} =====")

            backtest_results[asset] = {}

            for name, strategy in STRATEGIES.items():

                print(f"Backtesting strategy: {name}")

                strategy_data = strategy(df)

                result, equity_df = self.backtester.run(strategy_data)

                backtest_results[asset][name] = result

                plot_equity_curve(
                    equity_df,
                    title=f"{asset}_{name}"
                )

        return backtest_results

    def run(self):

        data = self.prepare_data()

        results = {}

        analysis_results = {}
        backtests = {}
        grid_results_all = {}

        for asset, df in data.items():

            print(f"\n==============================")
            print(f"RUNNING RESEARCH FOR {asset}")
            print(f"==============================")

            # ANALYSIS
            analysis_results[asset] = {}

            for name, experiment in EXPERIMENTS.items():

                print(f"Running experiment: {name}")

                analysis_results[asset][name] = experiment(df)
            
            # HORIZON ANALYSIS
            print(f"Running horizon analysis for {asset}")

            horizon_results = horizon_analysis(df)

            print(horizon_results)

            plot_horizon_analysis(horizon_results, asset)

            print(f"Running conditional horizon analysis for {asset}")

            # create volatility regimes
            df = df.copy()

            df["volatility_regime"] = pd.qcut(
                df["daily_range"],
                q=4,
                labels=[0,1,2,3]
            )

            cond_results = conditional_horizon_analysis(
                df,
                condition_col="volatility_regime"
            )   

            print(cond_results)

            plot_conditional_horizon(
                cond_results,
                asset,
                "volatility_regime"
            )

            # BACKTESTS
            backtests[asset] = self.run_backtests({asset: df})[asset]

            # GRID SEARCH FOR ALL STRATEGIES

            grid_results_all[asset] = {}

            for strategy_name, strategy_func in STRATEGIES.items():

                print(f"\nRunning grid search for {strategy_name}")

                param_space = PARAMETER_SPACE.get(strategy_name)

                if param_space is None:
                    print("No parameter space defined, skipping.")
                    continue

                grid_results = run_grid_search_parallel(
                    strategy_name,
                    strategy_func,
                    df,
                    self.backtester,
                    param_space
                )

                ranked = rank_results(grid_results)

                grid_results_all[asset][strategy_name] = ranked[:10]

                # Heatmap nur für volatility strategy
                if strategy_name == "volatility_trend_strategy":
                    plot_parameter_heatmap(grid_results)

            # WALK FORWARD
            wf = WalkForwardEngine(
                grid_engine=None,
                backtester=self.backtester
            )

            wf_results = wf.run(
            df,
            strategy_name=strategy_name,
            param_space=param_space
            )

            print(f"\nWalk-Forward results for {strategy_name}")
            print(wf_results)

            plot_walkforward_equity(wf_results)

            print(wf_results)

            plot_walkforward_equity(wf_results)

            print("Walk-forward mean Sharpe:", wf_results["sharpe"].mean())
            print("Walk-forward median Sharpe:", wf_results["sharpe"].median())
            print("Walk-forward total return:", (1 + wf_results["return"]).prod() - 1)

        results = {
            "analysis": analysis_results,
            "backtests": backtests,
            "grid_search": grid_results_all
        }

        self.save_results(results)

        return results
    

    def save_results(self, results):

        # ensure output directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = "results"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/experiment_{timestamp}.json"

        with open(filename, "w") as f:
            # some of the objects in results (e.g. datetimes) may not be
            # JSON-serializable by default.  `default=str` will convert
            # those to strings instead of raising TypeError.
            json.dump(results, f, indent=4, default=str)

        print(f"Results saved to {filename}")

