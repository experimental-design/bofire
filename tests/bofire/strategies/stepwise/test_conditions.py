import pytest

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.single import Himmelblau


def test_RequiredExperimentsCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(3), return_complete=True)
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
    "n_required, n_experiments, expected", [(1, 10, True), (2, 1, True)]
)
def test_CombiCondition(n_required, n_experiments, expected):
    benchmark = Himmelblau()
    experiments = benchmark.f(
        benchmark.domain.inputs.sample(n_experiments), return_complete=True
    )
    condition = data_models.CombiCondition(
        conditions=[
            data_models.NumberOfExperimentsCondition(n_experiments=2),
            data_models.NumberOfExperimentsCondition(n_experiments=12),
        ],
        n_required_conditions=n_required,
    )
    assert condition.evaluate(benchmark.domain, experiments=experiments) is expected
