import itertools

from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm

def generate_param_grid(param_space):

    keys = list(param_space.keys())
    values = list(param_space.values())

    combinations = list(itertools.product(*values))

    grid = []

    for combo in combinations:

        params = dict(zip(keys, combo))
        grid.append(params)

    return grid

def run_grid_search(strategy_name, strategy_func, data, backtester, param_space):

    from itertools import product

    keys = list(param_space.keys())
    values = list(param_space.values())

    results = []

    for combo in product(*values):

        params = dict(zip(keys, combo))

        strategy_returns = strategy_func(data, **params)

        metrics = backtester.run(strategy_returns)

        results.append({
            "strategy": strategy_name,
            "params": params,
            "metrics": metrics
        })

    return results

def rank_results(results, metric="sharpe"):

    ranked = sorted(
        results,
        key=lambda x: x["metrics"][metric],
        reverse=True
    )

    return ranked

def run_single_backtest(args):

    strategy_name, strategy_func, data, backtester, params = args

    strategy_returns = strategy_func(data, **params)

    metrics, _ = backtester.run(strategy_returns)

    return {
        "strategy": strategy_name,
        "params": params,
        "metrics": metrics
    }

def run_grid_search_parallel(strategy_name, strategy_func, data, backtester, param_space):

    keys = list(param_space.keys())
    values = list(param_space.values())

    param_combinations = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    tasks = [
        (strategy_name, strategy_func, data, backtester, params)
        for params in param_combinations
    ]

    workers = max(cpu_count() - 1, 1)

    print("First parameter set:", tasks[0][-1])
    print("Last parameter set:", tasks[-1][-1])
    print("Total combinations:", len(tasks))

    print(f"Running grid search with {workers} workers")

    with Pool(workers) as pool:

        results = []

        for result in tqdm(
            pool.imap(run_single_backtest, tasks),
            total=len(tasks),
            desc="Grid Search"
        ):
            results.append(result)

    return results