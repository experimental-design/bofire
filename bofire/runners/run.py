import json
import os
from copy import deepcopy
from typing import Callable, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain
from bofire.strategies.api import Strategy


class StrategyFactory(Protocol):
    def __call__(self, domain: Domain) -> Strategy: ...


def _single_run(
    run_idx: int,
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain, pd.DataFrame], float],
    n_candidates_per_proposals: int,
    safe_interval: int,
    initial_sampler: Optional[
        Union[Callable[[Domain], pd.DataFrame], pd.DataFrame]
    ] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    def autosafe_results(benchmark):
        """Safes results into a .json file to prevent data loss during
        time-expensive optimization runs. Autosave should operate every 10
        iterations.

        Args:
            benchmark: Benchmark function that is supposed be evaluated.

        """
        benchmark_name = benchmark.__class__.__name__
        # Create a folder for autosaves, if not already exists.
        if not os.path.exists("bofire_autosaves/" + benchmark_name):
            os.makedirs("bofire_autosaves/" + benchmark_name)

        filename = (
            "bofire_autosaves/" + benchmark_name + "/run" + str(run_idx) + ".json"
        )
        parsed_domain = benchmark.domain.model_dump_json()
        with open(filename, "w") as file:
            json.dump(parsed_domain, file)

    strategy = strategy_factory(domain=benchmark.domain)
    # sample initial values and tell
    if initial_sampler is not None:
        if isinstance(initial_sampler, Callable):
            X = initial_sampler(benchmark.domain)
            XY = benchmark.f(X, return_complete=True)
        else:
            XY = initial_sampler
        strategy.tell(XY)

    metric_values = np.zeros(n_iterations)
    pbar = tqdm(range(n_iterations), position=run_idx)
    for i in pbar:
        X = strategy.ask(candidate_count=n_candidates_per_proposals)
        X = X[benchmark.domain.inputs.get_keys()]
        Y = benchmark.f(X)
        XY = pd.concat([X, Y], axis=1)
        # pd.concat() changes datatype of str to np.int32 if column contains whole numbers.
        # column needs to be converted back to str to be added to the benchmark domain.
        strategy.tell(XY)
        metric_values[i] = metric(strategy.domain, strategy.experiments)
        pbar.set_description(f"Run {run_idx}")
        pbar.set_postfix({"Current Best:": f"{metric_values[i]:0.3f}"})
        if (i + 1) % safe_interval == 0:
            autosafe_results(benchmark=benchmark)
    return strategy.experiments, pd.Series(metric_values)


def run(
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain, pd.DataFrame], float],
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]] = None,
    n_candidates_per_proposal: int = 1,
    n_runs: int = 5,
    n_procs: int = 5,
    safe_interval: int = 1000,
) -> List[Tuple[pd.DataFrame, pd.Series]]:
    """Run a benchmark problem several times in parallel

    Args:
        benchmark: problem to be benchmarked
        strategy_factory: creates the strategy to be benchmarked on the benchmark
            problem
        n_iterations: number of times the strategy is asked
        metric: measure of success, e.g, best value found so far for single
            objective or hypervolume for multi-objective
        initial_sampler: Creates initial data
        n_candidates: also known as batch size, number of proposals made at once
            by the strategy
        n_runs: number of runs
        n_procs: number of parallel processes to execute the runs

    Returns:
        per run, a tuple with the benchmark object containing the proposed data and metric values

    """

    def make_args(run_idx: int):
        return (
            run_idx,
            deepcopy(benchmark),
            strategy_factory,
            n_iterations,
            metric,
            n_candidates_per_proposal,
            safe_interval,
            initial_sampler,
        )

    if n_procs == 1:
        results = [_single_run(*make_args(i)) for i in range(n_runs)]
    else:
        p = Pool(min(n_procs, n_runs))
        results = [p.apply_async(_single_run, make_args(i)) for i in range(n_runs)]
        results = [r.get() for r in results]
    return results
