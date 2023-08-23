import pytest

import bofire.data_models.strategies.api as data_models
import bofire.strategies.stepwise.conditions as conditions
from bofire.benchmarks.single import Himmelblau


def test_RequiredExperimentsCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(3), return_complete=True)
    data_model = data_models.NumberOfExperimentsCondition(n_experiments=3)
    condition = conditions.map(data_model=data_model)
    assert condition.evaluate(benchmark.domain, experiments=experiments) is True
    experiments = benchmark.f(benchmark.domain.inputs.sample(10), return_complete=True)
    assert condition.evaluate(benchmark.domain, experiments=experiments) is False


def test_RequiredExperimentsCondition_no_experiments():
    data_model = data_models.NumberOfExperimentsCondition(n_experiments=3)
    condition = conditions.map(data_model=data_model)
    assert condition.evaluate(Himmelblau().domain, experiments=None) is True


def test_AlwaysTrueCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(3), return_complete=True)
    data_model = data_models.AlwaysTrueCondition()
    condition = conditions.map(data_model=data_model)
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
    "n_required, n_experiments, expected", [(1, 10, True), (2, 1, True)]
)
def test_CombiCondition(n_required, n_experiments, expected):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(n_experiments), return_complete=True
    )
    data_model = data_models.CombiCondition(
        conditions=[
            data_models.NumberOfExperimentsCondition(n_experiments=2),
            data_models.NumberOfExperimentsCondition(n_experiments=12),
        ],
        n_required_conditions=n_required,
    )
    condition = conditions.map(data_model=data_model)
    assert condition.evaluate(benchmark.domain, experiments=experiments) is expected
