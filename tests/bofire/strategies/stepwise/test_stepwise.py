from copy import deepcopy

import pytest

import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.api import qNEI
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
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
                    max_parallelism=-1,
                ),
                Step(
                    strategy_data=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=15),
                    max_parallelism=2,
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
                    max_parallelism=-1,
                ),
                Step(
                    strategy_data=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                    max_parallelism=2,
                ),
            ],
        )


@pytest.mark.parametrize(
    "n_experiments, expected_strategy, expected_index",
    [(5, RandomStrategy, 0), (10, SoboStrategy, 1)],
)
def test_StepWiseStrategy_get_step(n_experiments, expected_strategy, expected_index):
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
                max_parallelism=-1,
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments)
    i, step = strategy._get_step()
    assert isinstance(step.strategy_data, expected_strategy)
    assert i == expected_index


def test_StepWiseStrategy_get_step_invalid():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(12), return_complete=True)
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=6),
                max_parallelism=-1,
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments)
    with pytest.raises(ValueError, match="No condition could be satisfied."):
        strategy._get_step()


def test_StepWiseStrategy_invalid_ask():
    benchmark = Himmelblau()
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=8),
                max_parallelism=2,
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    experiments = benchmark.f(benchmark.domain.inputs.sample(2), return_complete=True)
    strategy.tell(experiments=experiments)
    with pytest.raises(
        ValueError, match="Maximum number of candidates for step 0 is 2."
    ):
        strategy.ask(3)


def test_StepWiseStrategy_ask():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(2), return_complete=True)
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=benchmark.domain),
                condition=NumberOfExperimentsCondition(n_experiments=5),
                max_parallelism=2,
            ),
            Step(
                strategy_data=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments=experiments)
    candidates = strategy.ask(2)
    assert len(candidates) == 2
