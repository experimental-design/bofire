from copy import deepcopy

import pytest

import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.api import qEI, qNEI
from bofire.data_models.strategies.api import (
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)


def test_StepwiseStrategy_invalid():
    benchmark = Himmelblau()
    with pytest.raises(ValueError, match="First step has to be always applicable."):
        StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    data_model=RandomStrategy(domain=benchmark.domain),
                    num_required_experiments=1,
                    max_parallelism=-1,
                ),
                Step(
                    data_model=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    num_required_experiments=10,
                    max_parallelism=2,
                ),
            ],
        )
    with pytest.raises(
        ValueError, match="Step 2 needs less experiments than step 1. Wrong order."
    ):
        StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    data_model=RandomStrategy(domain=benchmark.domain),
                    num_required_experiments=0,
                    max_parallelism=-1,
                ),
                Step(
                    data_model=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    num_required_experiments=10,
                    max_parallelism=2,
                ),
                Step(
                    data_model=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qEI()
                    ),
                    num_required_experiments=5,
                    max_parallelism=2,
                ),
            ],
        )
    domain2 = deepcopy(benchmark.domain)
    domain2.inputs[0].key = "mama"
    with pytest.raises(
        ValueError,
        match="Domain of step 0 is incompatible to domain of StepwiseStrategy.",
    ):
        StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    data_model=RandomStrategy(domain=domain2),
                    num_required_experiments=0,
                    max_parallelism=-1,
                ),
                Step(
                    data_model=SoboStrategy(
                        domain=benchmark.domain, acquisition_function=qNEI()
                    ),
                    num_required_experiments=10,
                    max_parallelism=2,
                ),
            ],
        )


@pytest.mark.parametrize(
    "num_experiments, expected_strategy, expected_index",
    [(5, RandomStrategy, 0), (10, SoboStrategy, 1)],
)
def test_StepWiseStrategy_get_step(num_experiments, expected_strategy, expected_index):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(num_experiments), return_complete=True
    )
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                data_model=RandomStrategy(domain=benchmark.domain),
                num_required_experiments=0,
                max_parallelism=-1,
            ),
            Step(
                data_model=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                num_required_experiments=10,
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    strategy.tell(experiments)
    i, step = strategy._get_step()
    assert isinstance(step.data_model, expected_strategy)
    assert i == expected_index


def test_StepWiseStrategy_invalid_ask():
    benchmark = Himmelblau()
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                data_model=RandomStrategy(domain=benchmark.domain),
                num_required_experiments=0,
                max_parallelism=2,
            ),
            Step(
                data_model=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                num_required_experiments=10,
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    with pytest.raises(
        ValueError, match="Maximum number of candidates for step 0 is 2."
    ):
        strategy.ask(3)


def test_StepWiseStrategy_ask():
    benchmark = Himmelblau()
    data_model = StepwiseStrategy(
        domain=benchmark.domain,
        steps=[
            Step(
                data_model=RandomStrategy(domain=benchmark.domain),
                num_required_experiments=0,
                max_parallelism=2,
            ),
            Step(
                data_model=SoboStrategy(
                    domain=benchmark.domain, acquisition_function=qNEI()
                ),
                num_required_experiments=10,
                max_parallelism=2,
            ),
        ],
    )
    strategy = strategies.map(data_model)
    candidates = strategy.ask(2)
    assert len(candidates) == 2
