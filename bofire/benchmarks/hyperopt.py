import warnings
from typing import Optional, Tuple

import pandas as pd

import bofire.strategies.api as strategies
import bofire.surrogates.api as surrogates
from bofire.benchmarks.benchmark import Benchmark, run
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.strategies.api import (
    FactorialStrategy,
    RandomStrategy,
    SoboStrategy,
)
from bofire.data_models.surrogates.api import AnyTrainableSurrogate


class Hyperopt(Benchmark):
    def __init__(
        self,
        surrogate_data: AnyTrainableSurrogate,
        training_data: pd.DataFrame,
        folds: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        if surrogate_data.hyperconfig is None:
            raise ValueError("No hyperoptimization configuration found.")
        self.surrogate_data = surrogate_data
        self.training_data = training_data
        self.folds = folds
        self.results = None
        self.random_state = random_state

    @property
    def domain(self) -> Domain:
        return self.surrogate_data.hyperconfig.domain  # type: ignore

    @property
    def target_metric(self):
        return self.surrogate_data.hyperconfig.target_metric  # type: ignore

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        for i, candidate in candidates.iterrows():
            self.surrogate_data.update_hyperparameters(candidate)
            surrogate = surrogates.map(self.surrogate_data)
            _, cv_test, _ = surrogate.cross_validate(  # type: ignore
                self.training_data, folds=self.folds, random_state=self.random_state
            )
            if i == 0:
                results = cv_test.get_metrics(combine_folds=True)
            else:
                results = pd.concat(
                    [results, cv_test.get_metrics(combine_folds=True)],  # type: ignore
                    ignore_index=True,
                    axis=0,
                )
        results[f"valid_{self.target_metric.value}"] = 1  # type: ignore
        return results  # type: ignore


def hyperoptimize(
    surrogate_data: AnyTrainableSurrogate,
    training_data: pd.DataFrame,
    folds: int,
    random_state: Optional[int] = None,
) -> Tuple[AnyTrainableSurrogate, pd.DataFrame]:
    if surrogate_data.hyperconfig is None:
        warnings.warn(
            "No hyperopt is possible as no hyperopt config is available. Returning initial config."
        )
        return surrogate_data, pd.DataFrame({e.name: [] for e in RegressionMetricsEnum})

    def best(domain: Domain, experiments: pd.DataFrame) -> float:
        return (
            experiments[domain.outputs[0].key].min()
            if isinstance(domain.outputs[0].objective, MinimizeObjective)
            else experiments[domain.outputs[0].key].max()
        )

    def sample(domain):
        datamodel = RandomStrategy(domain=domain)
        sampler = strategies.map(data_model=datamodel)
        sampled = sampler.ask(len(domain.inputs) + 1)
        return sampled

    benchmark = Hyperopt(
        surrogate_data=surrogate_data,
        training_data=training_data,
        folds=folds,
        random_state=random_state,
    )

    if surrogate_data.hyperconfig.hyperstrategy == "FactorialStrategy":  # type: ignore
        strategy = strategies.map(FactorialStrategy(domain=benchmark.domain))
        experiments = benchmark.f(
            strategy.ask(candidate_count=None), return_complete=True
        )
    else:
        experiments = run(
            benchmark=benchmark,
            strategy_factory=RandomStrategy
            if surrogate_data.hyperconfig.hyperstrategy == "RandomStrategy"  # type: ignore
            else SoboStrategy,  # type: ignore
            metric=best,
            n_runs=1,
            n_iterations=surrogate_data.hyperconfig.n_iterations  # type: ignore
            - len(benchmark.domain.inputs)
            - 1,
            initial_sampler=sample,
            n_procs=1,
        )[0][0]

    # analyze the results and get the best
    experiments = experiments.sort_values(
        by=benchmark.target_metric.name,
        ascending=True
        if isinstance(benchmark.domain.outputs[0].objective, MinimizeObjective)
        else False,
    )

    surrogate_data.update_hyperparameters(experiments.iloc[0])

    return (
        surrogate_data,
        experiments[
            surrogate_data.hyperconfig.domain.inputs.get_keys()
            + [e.name for e in RegressionMetricsEnum]
        ],
    )
