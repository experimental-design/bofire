import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.constraints.api import LinearEqualityConstraint
from bofire.data_models.domain.api import Constraints, Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MaximizeSigmoidObjective,
)


def test_FeasibleExperimentCondition():
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
        ],
        outputs=[
            ContinuousOutput(key="of1", objective=MaximizeObjective()),
            ContinuousOutput(
                key="of2", objective=MaximizeSigmoidObjective(tp=5, steepness=1000)
            ),
            ContinuousOutput(
                key="of3", objective=MaximizeSigmoidObjective(tp=10, steepness=1000)
            ),
        ],
    )

    experiments = pd.DataFrame(
        {
            "x1": [0.1, 0.4, 0.7],
            "x2": [0.5, -2, 10],
            "of1": [0.1, 0.4, 0.7],
            "of2": [6, -2, 10],
            "of3": [11, 5.1, 0],
            "valid_of1": [True, True, True],
            "valid_of2": [True, True, True],
            "valid_of3": [True, True, True],
        }
    )

    condition = data_models.FeasibleExperimentCondition(
        n_required_feasible_experiments=3,
        threshold=0.9,
    )
    assert condition.evaluate(domain, experiments=experiments) is True
    condition.n_required_feasible_experiments = 2
    assert condition.evaluate(domain, experiments=experiments) is True
    assert condition.evaluate(domain, experiments=None) is True
    condition.n_required_feasible_experiments = 1
    assert condition.evaluate(domain, experiments=experiments) is False

    domain.constraints = Constraints(
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2"],
                coefficients=[
                    1.0,
                    1.0,
                ],
                rhs=200,
            )
        ]
    )
    assert condition.evaluate(domain, experiments=experiments) is True

    for feat in domain.outputs.get():
        feat.objective = MaximizeObjective()
    condition.n_required_feasible_experiments = 3
    assert condition.evaluate(domain, experiments=experiments) is False


def test_RequiredExperimentsCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(2), return_complete=True)
    condition = data_models.NumberOfExperimentsCondition(n_experiments=3)
    assert condition.evaluate(benchmark.domain, experiments=experiments) is True
    experiments = benchmark.f(benchmark.domain.inputs.sample(10), return_complete=True)
    assert condition.evaluate(benchmark.domain, experiments=experiments) is False


def test_RequiredExperimentsCondition_no_experiments():
    condition = data_models.NumberOfExperimentsCondition(n_experiments=3)
    assert condition.evaluate(Himmelblau().domain, experiments=None) is True


def test_AlwaysTrueCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(3), return_complete=True)
    condition = data_models.AlwaysTrueCondition()
    assert condition.evaluate(benchmark.domain, experiments=experiments) is True


def test_CombiCondition_invalid():
    with pytest.raises(
        ValueError,
        match="Number of required conditions larger than number of conditions.",
    ):
        data_models.CombiCondition(
            conditions=[
                data_models.NumberOfExperimentsCondition(n_experiments=2),
                data_models.NumberOfExperimentsCondition(n_experiments=3),
            ],
            n_required_conditions=3,
        )


@pytest.mark.parametrize(
    "n_required, n_experiments, expected",
    [(1, 10, True), (2, 1, True)],
)
def test_CombiCondition(n_required, n_experiments, expected):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(n_experiments),
        return_complete=True,
    )
    condition = data_models.CombiCondition(
        conditions=[
            data_models.NumberOfExperimentsCondition(n_experiments=2),
            data_models.NumberOfExperimentsCondition(n_experiments=12),
        ],
        n_required_conditions=n_required,
    )
    assert condition.evaluate(benchmark.domain, experiments=experiments) is expected
