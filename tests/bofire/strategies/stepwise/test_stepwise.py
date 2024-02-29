from copy import deepcopy
from typing import cast

import pandas as pd
import pytest

import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.api import qNEI
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    DropDataTransform,
    NumberOfExperimentsCondition,
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)


def test_StepwiseStrategy_invalid_domains():
    benchmark = Himmelblau()
    domain2 = deepcopy(benchmark.domain)
    domain2.inputs[0].key = "mama"
    with pytest.raises(
        ValueError,
        match="Domain of step 0 is incompatible to domain of StepwiseStrategy.",
        # match=f"Domain of step {0} is incompatible to domain of StepwiseStrategy.",
    ):
        StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    strategy_data=RandomStrategy(domain=domain2),
                    condition=NumberOfExperimentsCondition(n_experiments=5),
                ),
                Step(
                    strategy_data=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=15),
                ),
            ],
        )


def test_StepwiseStrategy_invalid_AlwaysTrue():
    benchmark = Himmelblau()
    with pytest.raises(
        ValueError, match="`AlwaysTrueCondition` is only allowed for the last step."
    ):
        StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    strategy_data=RandomStrategy(domain=benchmark.domain),
                    condition=AlwaysTrueCondition(),
                ),
                Step(
                    strategy_data=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
            ],
        )


@pytest.mark.parametrize(
    "n_experiments, expected_strategy",
    [(5, strategies.RandomStrategy), (10, strategies.SoboStrategy)],
)
def test_StepWiseStrategy_get_step(n_experiments, expected_strategy):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(n_experiments), return_complete=True
    )
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=6),
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
    strategy.tell(experiments)
    strategy, transform = strategy._get_step()
    assert transform is None
    assert isinstance(strategy, expected_strategy)


def test_StepWiseStrategy_get_step_invalid():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(12), return_complete=True)
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=6),
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
    strategy.tell(experiments)
    with pytest.raises(ValueError, match="No condition could be satisfied."):
        strategy._get_step()


def test_StepWiseStrategy_ask():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(2), return_complete=True)
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=5),
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments=experiments)
    candidates = strategy.ask(2)
    assert len(candidates) == 2


def test_remove_transform():
    benchmark = Himmelblau()
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=2),
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=AlwaysTrueCondition(),
                transform=DropDataTransform(to_be_removed_experiments=[0, 1]),
            ),
        ],
    )
    strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
    n_samples = 4
    experiments = pd.concat(
        [
            benchmark.domain.inputs.sample(n_samples),
            pd.DataFrame({"y": [1] * n_samples}),
        ],
        axis=1,
    )
    strategy.tell(experiments=experiments)
    for _ in range(2):
        x = strategy.ask()
        xy = pd.concat([x, pd.DataFrame({"y": [2]})], axis=1)
        strategy.tell(experiments=xy)
    last_strategy, _ = strategy._get_step()
    assert last_strategy.experiments is not None and len(last_strategy.experiments) == 3
