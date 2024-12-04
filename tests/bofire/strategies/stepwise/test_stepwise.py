from copy import deepcopy
from typing import cast

import pytest

import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.api import qNEI
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    NumberOfExperimentsCondition,
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)
from bofire.data_models.strategies.stepwise.stepwise import (
    validate_domain_compatibility,
)


def test_validate_domain_compatibility():
    bench = Himmelblau()

    domain2 = deepcopy(bench.domain)
    domain2.inputs = bench.domain.inputs.get_by_keys(["x_1"])
    domain2.inputs.features.append(CategoricalInput(key="x_2", categories=["a", "b"]))
    with pytest.raises(ValueError, match="Features with key x_2 have different types."):
        validate_domain_compatibility(bench.domain, domain2)

    domain2 = deepcopy(bench.domain)
    domain2.inputs.features.append(CategoricalInput(key="x_3", categories=["a", "b"]))
    domain3 = deepcopy(bench.domain)
    domain3.inputs.features.append(CategoricalInput(key="x_3", categories=["a", "c"]))
    with pytest.raises(
        ValueError, match="Features with key x_3 have different categories"
    ):
        validate_domain_compatibility(domain2, domain3)


def test_StepwiseStrategy_invalid_domains():
    benchmark = Himmelblau()
    domain2 = deepcopy(benchmark.domain)
    domain2.inputs.features[0] = CategoricalInput(key="x_1", categories=["a", "b"])
    with pytest.raises(
        ValueError,
        match="Features with key x_1 have different types.",
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
                        domain=benchmark.domain,
                        acquisition_function=qNEI(),
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=15),
                ),
            ],
        )


def test_StepwiseStrategy_invalid_AlwaysTrue():
    benchmark = Himmelblau()
    with pytest.raises(
        ValueError,
        match="`AlwaysTrueCondition` is only allowed for the last step.",
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
                        domain=benchmark.domain,
                        acquisition_function=qNEI(),
                    ),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
            ],
        )


@pytest.mark.parametrize(
    "n_experiments, expected_strategy",
    [(5, strategies.RandomStrategy), (9, strategies.SoboStrategy)],
)
def test_StepWiseStrategy_get_step(n_experiments, expected_strategy):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(n_experiments),
        return_complete=True,
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
                    domain=benchmark.domain,
                    acquisition_function=qNEI(),
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
    strategy.tell(experiments)
    _strategy, transform = strategy.get_step()
    assert transform is None
    assert isinstance(_strategy, expected_strategy)
    if isinstance(_strategy, strategies.RandomStrategy):
        with pytest.raises(ValueError):
            _ = strategy.surrogates
        with pytest.raises(ValueError):
            _ = strategy.surrogates_specs
    else:
        assert strategy.surrogates == _strategy.surrogates  # type: ignore
        assert strategy.surrogates_specs == _strategy.surrogate_specs  # type: ignore


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
                    domain=benchmark.domain,
                    acquisition_function=qNEI(),
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
    strategy.tell(experiments)
    with pytest.raises(ValueError, match="No condition could be satisfied."):
        strategy.get_step()


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
                    domain=benchmark.domain,
                    acquisition_function=qNEI(),
                ),
                condition=NumberOfExperimentsCondition(n_experiments=10),
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments=experiments)
    candidates = strategy.ask(2)
    assert len(candidates) == 2
