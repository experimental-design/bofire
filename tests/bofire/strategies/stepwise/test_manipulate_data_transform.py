from copy import deepcopy

from pandas.testing import assert_frame_equal, assert_series_equal

import bofire.strategies.api as strategies
import bofire.transforms.api as transforms
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.strategies.predictives.sobo import SoboStrategy
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy
from bofire.data_models.transforms.api import ManipulateDataTransform


def test_dropdata_transform():
    bench = Himmelblau()
    candidates = bench.domain.inputs.sample(10)
    experiments = bench.f(bench.domain.inputs.sample(10), return_complete=True)

    transform_data = ManipulateDataTransform(
        experiment_transforms=["x_1 = x_1 + 100", "x_2 = x_2  / 2.0"],
        candidate_transforms=["x_1 = x_1 -20", "x_2 = x_2  / 2.0"],
        candidate_untransforms=["x_1 = x_1 + 20", "x_2 = x_2 * 2.0"],
    )

    transform = transforms.map(transform_data)

    transformed_experiments = transform.transform_experiments(experiments)
    transformed_candidates = transform.transform_candidates(candidates)
    untransformed_candidates = transform.untransform_candidates(transformed_candidates)

    assert_series_equal(experiments.x_1 + 100, transformed_experiments.x_1)
    assert_series_equal(experiments.x_2 / 2.0, transformed_experiments.x_2)

    try:
        assert_frame_equal(candidates, transformed_candidates)
    except AssertionError:
        pass

    assert_frame_equal(candidates, untransformed_candidates)


def test_stepwise():
    bench = Himmelblau()
    candidates = bench.domain.inputs.sample(10)

    transform_data = ManipulateDataTransform(
        candidate_untransforms=["x_1 = x_1 + 200", "x_2 = x_2 - 200"],
    )

    domain = deepcopy(bench.domain)
    domain.inputs.get_by_key("x_1").bounds = (-6, 300)
    domain.inputs.get_by_key("x_2").bounds = (-300, 6)
    strategy_data = StepwiseStrategy(
        domain=domain,
        steps=[
            Step(
                condition=NumberOfExperimentsCondition(n_experiments=5),
                strategy_data=RandomStrategy(domain=bench.domain),
                transform=transform_data,
            ),
            Step(
                condition=AlwaysTrueCondition(),
                strategy_data=SoboStrategy(domain=bench.domain),
            ),
        ],
    )

    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(candidate_count=1)
    assert all(candidates.x_1 >= 150)
    assert all(candidates.x_2 <= -150)
