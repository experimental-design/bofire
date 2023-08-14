import json
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
from multiprocess.pool import Pool
from pydantic import Field, PositiveFloat
from scipy.stats import norm, uniform
from tqdm import tqdm
from typing_extensions import Annotated

import bofire.strategies.api as strategies
from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import AnyStrategy


class OutlierPrior(BaseModel):
    type: str


class UniformOutlierPrior(OutlierPrior):
    type: Literal["UniformOutlierPrior"] = "UniformOutlierPrior"
    bounds: Tuple[float, float]

    def sample(self, n_samples: int) -> np.ndarray:
        return uniform(
            self.bounds[0],
            self.bounds[1] - self.bounds[0],
        ).rvs(n_samples)


class NormalOutlierPrior(OutlierPrior):
    type: Literal["NormalOutlierPrior"] = "NormalOutlierPrior"
    loc: float
    scale: PositiveFloat

    def sample(self, n_samples: int) -> np.ndarray:
        return norm(self.loc, self.scale).rvs(n_samples)


AnyOutlierPrior = Union[UniformOutlierPrior, NormalOutlierPrior]


class Benchmark:
    def __init__(
        self,
        outlier_rate: Annotated[float, Field(ge=0, lt=1)] = 0,
        outlier_prior: Optional[AnyOutlierPrior] = None,
    ):
        self.outlier_rate = outlier_rate
        self.outlier_prior = outlier_prior

    def f(
        self,
        candidates: pd.DataFrame,
        return_complete: bool = False,
    ) -> pd.DataFrame:
        Y = self._f(candidates)
        if self.outlier_prior is not None:
            for output_feature in self.domain.outputs.get_keys():
                # no_outliers = int(len(Y) * self.outlier_rate)
                ix2 = np.zeros(len(Y), dtype=bool)
                ix1 = uniform().rvs(len(Y))
                # ix2[np.random.choice(len(Y), no_outliers, replace=False)] = True
                ix2 = ix1 <= self.outlier_rate
                n_outliers = sum(ix2)
                Y.loc[ix2, output_feature] = Y.loc[ix2, output_feature] + self.outlier_prior.sample(n_outliers)  # type: ignore
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
    def __call__(self, domain: Domain) -> AnyStrategy:
        ...


def _single_run(
    run_idx: int,
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain, pd.DataFrame], float],
    n_candidates_per_proposals: int,
    safe_intervall: int,
    initial_sampler: Optional[
        Union[Callable[[Domain], pd.DataFrame], pd.DataFrame]
    ] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
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
        parsed_domain = benchmark.domain.json()
        with open(filename, "w") as file:
            json.dump(parsed_domain, file)

    # sample initial values
    if initial_sampler is not None:
        if isinstance(initial_sampler, Callable):
            X = initial_sampler(benchmark.domain)
            XY = benchmark.f(X, return_complete=True)
        else:
            XY = initial_sampler
    strategy_data = strategy_factory(domain=benchmark.domain)
    # map it
    strategy = strategies.map(strategy_data)  # type: ignore
    # tell it
    if initial_sampler is not None:
        strategy.tell(XY)  # type: ignore
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
        metric_values[i] = metric(strategy.domain, strategy.experiments)  # type: ignore
        pbar.set_description(
            f"run {run_idx:02d} with current best {metric_values[i]:0.3f}"
        )
        if (i + 1) % safe_intervall == 0:
            autosafe_results(benchmark=benchmark)
    return strategy.experiments, pd.Series(metric_values)  # type: ignore


def run(
    benchmark: Benchmark,
    strategy_factory: StrategyFactory,
    n_iterations: int,
    metric: Callable[[Domain, pd.DataFrame], float],
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]] = None,
    n_candidates_per_proposal: int = 1,
    n_runs: int = 5,
    n_procs: int = 5,
    safe_intervall: int = 1000,
) -> List[Tuple[pd.DataFrame, pd.Series]]:
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
            n_candidates_per_proposal,
            safe_intervall,
            initial_sampler,
        )

    if n_procs == 1:
        results = [_single_run(*make_args(i)) for i in range(n_runs)]
    else:
        p = Pool(min(n_procs, n_runs))
        results = [p.apply_async(_single_run, make_args(i)) for i in range(n_runs)]
        results = [r.get() for r in results]
    return results
