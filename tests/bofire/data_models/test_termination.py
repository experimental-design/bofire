"""Tests for termination conditions data models."""

import pandas as pd
import pytest

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.termination.api import (
    AlwaysContinue,
    CombiTerminationCondition,
    MaxIterationsTermination,
    UCBLCBRegretTermination,
)


@pytest.fixture
def simple_domain():
    return Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
        ],
        outputs=[
            ContinuousOutput(key="y", objective=MaximizeObjective()),
        ],
    )


@pytest.fixture
def sample_experiments():
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "x2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [1.0, 2.0, 3.0, 2.5, 2.8],
        }
    )


class TestMaxIterationsTermination:
    def test_init(self):
        term = MaxIterationsTermination(max_iterations=100)
        assert term.max_iterations == 100
        assert term.type == "MaxIterationsTermination"

    def test_should_terminate_before_max(self, simple_domain, sample_experiments):
        term = MaxIterationsTermination(max_iterations=10)
        assert not term.should_terminate(simple_domain, sample_experiments, 5)

    def test_should_terminate_at_max(self, simple_domain, sample_experiments):
        term = MaxIterationsTermination(max_iterations=10)
        assert term.should_terminate(simple_domain, sample_experiments, 10)

    def test_should_terminate_after_max(self, simple_domain, sample_experiments):
        term = MaxIterationsTermination(max_iterations=10)
        assert term.should_terminate(simple_domain, sample_experiments, 15)


class TestAlwaysContinue:
    def test_init(self):
        term = AlwaysContinue()
        assert term.type == "AlwaysContinue"

    def test_never_terminates(self, simple_domain, sample_experiments):
        term = AlwaysContinue()
        for i in range(100):
            assert not term.should_terminate(simple_domain, sample_experiments, i)


class TestUCBLCBRegretTermination:
    def test_init_defaults(self):
        term = UCBLCBRegretTermination()
        assert term.noise_variance is None
        assert term.threshold_factor == 1.0
        assert term.min_iterations == 5

    def test_should_not_terminate_before_min_iterations(
        self, simple_domain, sample_experiments
    ):
        term = UCBLCBRegretTermination(min_iterations=10)
        assert not term.should_terminate(
            simple_domain, sample_experiments, 5, regret_bound=0.0
        )

    def test_should_terminate_when_regret_below_threshold(
        self, simple_domain, sample_experiments
    ):
        term = UCBLCBRegretTermination(noise_variance=0.1, min_iterations=3)
        assert term.should_terminate(
            simple_domain, sample_experiments, 5, regret_bound=0.01
        )

    def test_should_not_terminate_when_regret_above_threshold(
        self, simple_domain, sample_experiments
    ):
        term = UCBLCBRegretTermination(noise_variance=0.01, min_iterations=3)
        assert not term.should_terminate(
            simple_domain, sample_experiments, 5, regret_bound=1.0
        )

    def test_uses_estimated_noise_variance(self, simple_domain, sample_experiments):
        term = UCBLCBRegretTermination(min_iterations=3)
        assert term.should_terminate(
            simple_domain,
            sample_experiments,
            5,
            regret_bound=0.001,
            estimated_noise_variance=0.01,
        )


class TestCombiTerminationCondition:
    def test_init(self):
        term = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=100),
                AlwaysContinue(),
            ],
            n_required_conditions=1,
        )
        assert len(term.conditions) == 2
        assert term.n_required_conditions == 1

    def test_or_logic(self, simple_domain, sample_experiments):
        term = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=10),
                MaxIterationsTermination(max_iterations=20),
            ],
            n_required_conditions=1,
        )
        assert term.should_terminate(simple_domain, sample_experiments, 10)
        assert not term.should_terminate(simple_domain, sample_experiments, 5)

    def test_and_logic(self, simple_domain, sample_experiments):
        term = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=10),
                MaxIterationsTermination(max_iterations=20),
            ],
            n_required_conditions=2,
        )
        assert not term.should_terminate(simple_domain, sample_experiments, 15)
        assert term.should_terminate(simple_domain, sample_experiments, 20)

    def test_invalid_n_required_conditions(self):
        with pytest.raises(ValueError, match="cannot be larger"):
            CombiTerminationCondition(
                conditions=[MaxIterationsTermination(max_iterations=10)],
                n_required_conditions=5,
            )

    def test_nested_combination(self, simple_domain, sample_experiments):
        inner = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=10),
                MaxIterationsTermination(max_iterations=15),
            ],
            n_required_conditions=2,
        )
        outer = CombiTerminationCondition(
            conditions=[inner, MaxIterationsTermination(max_iterations=5)],
            n_required_conditions=1,
        )
        assert outer.should_terminate(simple_domain, sample_experiments, 5)


class TestTerminationConditionSerialization:
    def test_max_iterations_serialization(self):
        term = MaxIterationsTermination(max_iterations=100)
        data = term.model_dump()
        restored = MaxIterationsTermination(**data)
        assert restored.max_iterations == 100

    def test_ucblcb_serialization(self):
        term = UCBLCBRegretTermination(noise_variance=0.1, threshold_factor=2.0)
        data = term.model_dump()
        restored = UCBLCBRegretTermination(**data)
        assert restored.noise_variance == 0.1
        assert restored.threshold_factor == 2.0

    def test_combi_serialization(self):
        term = CombiTerminationCondition(
            conditions=[
                MaxIterationsTermination(max_iterations=100),
                UCBLCBRegretTermination(),
            ],
            n_required_conditions=1,
        )
        data = term.model_dump()
        assert len(data["conditions"]) == 2
