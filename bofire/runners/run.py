import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain
from bofire.strategies.api import Strategy


class StrategyFactory(Protocol):
    def __call__(self, domain: Domain) -> Strategy: ...


@dataclass
class RunResult:
    """Result of a single optimization run.

    Supports tuple-style access for backward compatibility with ``run()``,
    which historically returned ``List[Tuple[pd.DataFrame, pd.Series]]``.
    You can still write ``experiments, metric_values = result`` or
    ``result[0]`` / ``result[1]``.
    """

    experiments: pd.DataFrame
    metric_values: pd.Series
    terminated_early: bool = False
    final_iteration: int = 0
    termination_metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Backward-compat: behave like (experiments, metric_values) tuple.
    def __iter__(self):
        return iter((self.experiments, self.metric_values))

    def __getitem__(self, idx):
        return (self.experiments, self.metric_values)[idx]


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
    termination_condition: Any = None,
) -> RunResult:
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

    # Set up termination evaluators if a condition is provided
    evaluators: list = []
    if termination_condition is not None:
        from bofire.termination.api import get_required_evaluators

        evaluators = get_required_evaluators(termination_condition)

    metric_values: List[float] = []
    termination_metrics: Dict[str, List[float]] = {}
    terminated_early = False
    final_iteration = 0

    pbar = tqdm(range(n_iterations), position=run_idx)
    for i in pbar:
        X = strategy.ask(candidate_count=n_candidates_per_proposals)
        X = X[benchmark.domain.inputs.get_keys()]
        Y = benchmark.f(X)
        XY = pd.concat([X, Y], axis=1)
        # pd.concat() changes datatype of str to np.int32 if column contains whole numbers.
        # column needs to be converted back to str to be added to the benchmark domain.
        strategy.tell(XY)
        current_metric = metric(strategy.domain, strategy.experiments)
        metric_values.append(current_metric)

        # Check termination condition if provided
        if termination_condition is not None:
            eval_kwargs: Dict[str, Any] = {}
            for evaluator in evaluators:
                metrics = evaluator.evaluate(strategy, strategy.experiments, i)
                eval_kwargs.update(metrics)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in termination_metrics:
                            termination_metrics[key] = []
                        termination_metrics[key].append(value)

            if termination_condition.should_terminate(
                domain=strategy.domain,
                experiments=strategy.experiments,
                iteration=i,
                **eval_kwargs,
            ):
                terminated_early = True
                final_iteration = i
                break

        pbar.set_description(f"Run {run_idx}")
        pbar.set_postfix({"Current Best:": f"{current_metric:0.3f}"})
        if (i + 1) % safe_interval == 0:
            autosafe_results(benchmark=benchmark)

        final_iteration = i

    return RunResult(
        experiments=strategy.experiments,
        metric_values=pd.Series(metric_values),
        terminated_early=terminated_early,
        final_iteration=final_iteration,
        termination_metrics=termination_metrics,
    )


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
    termination_condition: Any = None,
) -> List[RunResult]:
    """Run a benchmark problem several times in parallel.

    Args:
        benchmark: problem to be benchmarked
        strategy_factory: creates the strategy to be benchmarked on the benchmark
            problem
        n_iterations: number of times the strategy is asked
        metric: measure of success, e.g, best value found so far for single
            objective or hypervolume for multi-objective
        initial_sampler: Creates initial data
        n_candidates_per_proposal: also known as batch size, number of proposals
            made at once by the strategy
        n_runs: number of runs
        n_procs: number of parallel processes to execute the runs
        safe_interval: Interval for autosaving results.
        termination_condition: Optional termination condition (from
            ``bofire.data_models.termination.api``). When provided, runs may
            stop early. Defaults to ``None`` (run all *n_iterations*).

    Returns:
        List of :class:`RunResult`, one per run.  Each ``RunResult`` can be
        unpacked as ``(experiments, metric_values)`` for backward
        compatibility.

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
            termination_condition,
        )

    if n_procs == 1:
        results = [_single_run(*make_args(i)) for i in range(n_runs)]
    else:
        p = Pool(min(n_procs, n_runs))
        results = [p.apply_async(_single_run, make_args(i)) for i in range(n_runs)]
        results = [r.get() for r in results]
    return results
