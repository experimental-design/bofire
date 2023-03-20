import json
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import AnyStrategy


class Benchmark:
    """
    A class for implementing benchmark functions and evaluating their performance.
    """

    def f(
        self, candidates: pd.DataFrame, return_complete: bool = False
    ) -> pd.DataFrame:
        """Evaluates the benchmark function on the input candidates and returns the output.

        Args:
            candidates (pd.DataFrame): The input candidates of the benchmark function that need to be evaluated.
            return_complete (bool, optional): If return_complete is set to True, the function returns a concatenated DataFrame with input and output values.
            Otherwise it returns the output values. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame with either the input and output values or just the output values.
        """
        Y = self._f(candidates)

        if return_complete:
            return pd.concat([candidates, Y], axis=1)

        return Y

    @abstractmethod
    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to be implemented by derived classes.
        Takes a DataFrame of input candidates and returns a DataFrame of corresponding function outputs.

        Args:
            candidates (pd.DataFrame): The input candidates of the benchmark function that need to be evaluated.

        Returns:
            pd.DataFrame: A DataFrame with either the input and output values or just the output values.
        """
        pass

    def get_optima(self) -> pd.DataFrame:
        """Raises a NotImplementedError. This method should be implemented by derived classes to return
        the optimal set of input candidates and their corresponding function outputs.

        Raises:
            NotImplementedError: _description_

        Returns:
            pd.DataFrame: The optimal set of input candidates and their corresponding function outputs
        """
        raise NotImplementedError()

    @property
    def domain(self) -> Domain:
        """A read-only property that returns the domain of the benchmark function.

        Returns:
            Domain: The domain of the benchmark function.
        """
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
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Runs a single iteration of an optimization algorithm and returns the results.

    Parameters:
    -----------
    run_idx: int
        The index of the current run.
    benchmark: Benchmark
        A function that takes a set of input parameters and returns a set of output values that are supposed to be optimized.
    strategy_factory: StrategyFactory
        A function that creates an optimization strategy for the given benchmark function.
    n_iterations: int
        The number of iterations to run the optimization algorithm.
    metric: Callable[[Domain, pd.DataFrame], float]
        A function that takes the optimization domain and a dataframe of experiments and returns a scalar value representing the quality of the optimization results.
    n_candidates_per_proposals: int
        The number of candidate solutions to generate per proposal.
    safe_intervall: int
        The frequency at which to perform autosaving of results.
    initial_sampler: Optional[Callable[[Domain], pd.DataFrame]] = None
        An optional function that generates an initial set of input parameters for the optimization algorithm.

    Returns:
    --------
    A tuple of two pandas objects:
    - A dataframe representing the set of experiments performed during the optimization algorithm.
    - A series representing the values of the optimization metric at each iteration.
    """

    def autosafe_results(benchmark: Benchmark):
        """A function that safes results into a .json file to prevent data loss during time-expensive optimization runs.
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
        X = initial_sampler(benchmark.domain)
        XY = benchmark.f(X, return_complete=True)
    strategy_data = strategy_factory(domain=benchmark.domain)
    # map it
    strategy = strategies.map(strategy_data)
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
        """
        make_args - Function that takes in an integer run_idx and returns a tuple of arguments required to run a benchmark optimization algorithm.

        Args:
        run_idx (int): Integer index of the current run.

        Returns:
        A tuple of arguments required to run a benchmark optimization algorithm:
        - run_idx (int): Integer index of the current run.
        - benchmark (object): A deepcopy of the benchmark object that specifies the objective function to be optimized.
        - strategy_factory (function): A function that returns an optimization strategy object that implements the optimization algorithm.
        - n_iterations (int): Number of iterations to run the optimization algorithm.
        - metric (str): Name of the metric used to evaluate the performance of the optimization algorithm.
        - n_candidates_per_proposal (int): Number of candidate solutions proposed by the optimization algorithm at each iteration.
        - safe_intervall (int): The minimum distance between consecutive solutions proposed by the optimization algorithm.
        - initial_sampler (object): An object that implements the initial sampling strategy for the optimization algorithm.

        Raises:
        None
        """
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
