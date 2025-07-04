import warnings
from typing import Optional, Tuple

import pandas as pd
from botorch.exceptions.warnings import InputDataWarning

import bofire.strategies.api as strategies
import bofire.strategies.mapper as strategy_mapper
from bofire.benchmarks.api import Hyperopt
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.strategies.api import (
    FractionalFactorialStrategy,
    RandomStrategy,
    SoboStrategy,
)
from bofire.data_models.surrogates.api import AnyTrainableSurrogate
from bofire.runners.run import run


# ignore warning related to evaluating test data using RobustSingleTaskGPSurrogate
warnings.filterwarnings("ignore", category=InputDataWarning)


def hyperoptimize(
    surrogate_data: AnyTrainableSurrogate,
    training_data: pd.DataFrame,
    folds: int,
    random_state: Optional[int] = None,
) -> Tuple[AnyTrainableSurrogate, pd.DataFrame]:
    if surrogate_data.hyperconfig is None:
        warnings.warn(
            "No hyperopt is possible as no hyperopt config is available. Returning initial config.",
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
        show_progress_bar=True
        if surrogate_data.hyperconfig.hyperstrategy == "FractionalFactorialStrategy"
        else False,
    )

    if surrogate_data.hyperconfig.hyperstrategy == "FractionalFactorialStrategy":
        strategy = strategies.map(FractionalFactorialStrategy(domain=benchmark.domain))
        experiments = benchmark.f(
            strategy.ask(candidate_count=None),
            return_complete=True,
        )
    else:
        strategy_data = (
            RandomStrategy
            if surrogate_data.hyperconfig.hyperstrategy == "RandomStrategy"
            else SoboStrategy
        )
        experiments = run(
            benchmark=benchmark,
            strategy_factory=lambda domain: strategy_mapper.map(
                data_model=strategy_data(domain=domain),
            ),
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
        ascending=(
            True
            if isinstance(benchmark.domain.outputs[0].objective, MinimizeObjective)
            else False
        ),
    )

    surrogate_data.update_hyperparameters(experiments.iloc[0])

    return (
        surrogate_data,
        experiments[
            surrogate_data.hyperconfig.domain.inputs.get_keys()
            + [e.name for e in RegressionMetricsEnum]
        ],
    )
