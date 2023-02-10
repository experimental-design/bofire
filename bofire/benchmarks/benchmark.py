import json
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from bofire.domain.domain import Domain
from bofire.domain.features import OutputFeature
from bofire.domain.objectives import Objective
from bofire.strategies.strategy import Strategy


# TODO: remove reduction parameter as soon as additive/multiplicative is part of Domain
def best(domain: Domain, reduction: Callable[[pd.DataFrame], pd.Series]):
    assert domain.experiments is not None
    outputs_with_objectives = domain.outputs.get_by_objective(Objective)
    output_values = domain.experiments[outputs_with_objectives.get_keys()]
    objective_values = list()
    for output, col_name in zip(outputs_with_objectives, output_values):
        assert isinstance(output, OutputFeature)
        assert output.objective is not None
        objective_values.append(output.objective(output_values[col_name]))
    objective_values = reduction(pd.concat([ov.to_frame() for ov in objective_values]))
    return objective_values.max()


# TODO: remove as soon as additive/multiplicative is part of Domain
def best_additive(domain: Domain):
    return best(domain, lambda df: df.sum(axis=1))


# TODO: remove as soon as additive/multiplicative is part of Domain
def best_multiplicative(domain: Domain):
    return best(domain, lambda df: df.prod(axis=1))


class Benchmark:
    def f(
        self, candidates: pd.DataFrame, return_complete: bool = False
    ) -> pd.DataFrame:
        Y = self._f(candidates)

        if return_complete:
            return pd.concat([candidates, Y], axis=1)

        return Y

    @abstractmethod
    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_optima(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def domain(self) -> Domain:
        return self._domain  # type: ignore


class StrategyFactory(Protocol):
    def __call__(self, domain: Domain) -> Strategy:
        ...


def _single_run(
    run_idx: int,
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain], float],
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]],
    n_candidates_per_proposals: int,
) -> Tuple[Benchmark, pd.Series]:
    def autosafe_results(benchmark):
        """Safes results into a .json file to prevent data loss during time-expensive optimization runs.
        Autosave should operate every 10 iterations.

        Args:
            benchmark: Benchmark function that is suposed be evaluated.
        """

        benchmark_name = benchmark.__class__.__name__
        # Create a folder for autosaves, if not already exists.
        if not os.path.exists("bofire_autosaves/" + benchmark_name):
            os.makedirs("bofire_autosaves/" + benchmark_name)

        filename = (
            "bofire_autosaves/" + benchmark_name + "/run" + str(run_idx) + ".json"
        )
        # TODO: Save domain as a whole into a .json. Possible when domain can be parsed into a dict (waiting for an update)
        # parsed_domain = benchmark.domain.dict()
        parsed_domain = benchmark.domain.experiments.to_json(orient="split")
        with open(filename, "w") as file:
            json.dump(parsed_domain, file)

    if initial_sampler is not None:
        X = initial_sampler(benchmark.domain)
        Y = benchmark.f(X)
        XY = pd.concat([X, Y], axis=1)
        benchmark.domain.add_experiments(XY)
    strategy = strategy_factory(domain=benchmark.domain)
    metric_values = np.zeros(n_iterations)
    pbar = tqdm(range(n_iterations), position=run_idx)
    for i in pbar:
        X = strategy.ask(candidate_count=n_candidates_per_proposals)
        X = X[benchmark.domain.inputs.get_keys()]
        Y = benchmark.f(X)
        XY = pd.concat([X, Y], axis=1)
        # pd.concat() changes datatype of str to np.int32 if column contains whole numbers.
        # colum needs to be converted back to str to be added to the benchmark domain.
        strategy.tell(XY)
        metric_values[i] = metric(strategy.domain)
        pbar.set_description(
            f"run {run_idx:02d} with current best {metric_values[i]:0.3f}"
        )
        if (i + 1) % 1 == 0:
            autosafe_results(benchmark=benchmark)
    return benchmark, pd.Series(metric_values)


def run(
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain], float],
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]] = None,
    n_candidates_per_proposal: int = 1,
    n_runs: int = 5,
    n_procs: int = 5,
) -> List[Tuple[Benchmark, pd.Series]]:
    """Run a benchmark problem several times in parallel

    Args:
        benchmark: problem to be benchmarked
        strategy_factory: creates the strategy to be benchmarked on the benchmark problem
        n_iterations: number of times the strategy is asked
        metric: measure of success, e.g, best value found so far for single objective or
                hypervolume for multi-objective
        initial_sampler: Creates initial data
        n_candidates: also known as batch size, number of proposals made at once by the strategy
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
            initial_sampler,
            n_candidates_per_proposal,
        )

    if n_procs == 1:
        results = [_single_run(*make_args(i)) for i in range(n_runs)]
    else:
        p = Pool(min(n_procs, n_runs))
        results = [p.apply_async(_single_run, make_args(i)) for i in range(n_runs)]
        results = [r.get() for r in results]
    return results
