import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.constraints.api import LinearEqualityConstraint
from bofire.data_models.domain.api import Constraints, Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import (
    ConstrainedCategoricalObjective,
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
    assert condition.evaluate(None, domain, experiments=experiments) is True
    condition.n_required_feasible_experiments = 2
    assert condition.evaluate(None, domain, experiments=experiments) is True
    assert condition.evaluate(None, domain, experiments=None) is True
    condition.n_required_feasible_experiments = 1
    assert condition.evaluate(None, domain, experiments=experiments) is False

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
    assert condition.evaluate(None, domain, experiments=experiments) is True

    for feat in domain.outputs.get():
        feat.objective = MaximizeObjective()
    condition.n_required_feasible_experiments = 3
    assert condition.evaluate(None, domain, experiments=experiments) is False


def test_RequiredExperimentsCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(2), return_complete=True)
    condition = data_models.NumberOfExperimentsCondition(n_experiments=3)
    assert condition.evaluate(None, benchmark.domain, experiments=experiments) is True
    experiments = benchmark.f(benchmark.domain.inputs.sample(10), return_complete=True)
    assert condition.evaluate(None, benchmark.domain, experiments=experiments) is False


def test_RequiredExperimentsCondition_no_experiments():
    condition = data_models.NumberOfExperimentsCondition(n_experiments=3)
    assert condition.evaluate(None, Himmelblau().domain, experiments=None) is True


def test_AlwaysTrueCondition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(3), return_complete=True)
    condition = data_models.AlwaysTrueCondition()
    assert condition.evaluate(None, benchmark.domain, experiments=experiments) is True


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
    assert (
        condition.evaluate(None, benchmark.domain, experiments=experiments) is expected
    )


def _sufficient_domain():
    return Domain(
        inputs=[
            ContinuousInput(key="x", bounds=[0, 10]),
            CategoricalInput(key="c", categories=["a", "b", "c"]),
        ],
        outputs=[
            ContinuousOutput(key="y1", objective=MaximizeObjective()),
            ContinuousOutput(key="y2", objective=MaximizeObjective()),
            CategoricalOutput(
                key="co",
                categories=["u", "v", "w"],
                objective=ConstrainedCategoricalObjective(
                    categories=["u", "v", "w"], desirability=[True, True, False]
                ),
            ),
        ],
    )


def _sufficient_condition(key, threshold, n_required):
    params = data_models.DistinguishableParameters(
        key=key, threshold=threshold, n_different_elements=n_required
    )
    if key in ("x", "c"):
        return data_models.SufficientDistinguishableValues(inputs=[params])
    return data_models.SufficientDistinguishableValues(outputs=[params])


def _sufficient_experiments(x, y1, y2=None, c=None, co=None, valid_co=None):
    return pd.DataFrame(
        {
            "x": x,
            "c": ["a"] * len(x) if c is None else c,
            "y1": y1,
            "y2": list(range(len(x))) if y2 is None else y2,
            "co": ["u"] * len(x) if co is None else co,
            "valid_y1": [True] * len(x),
            "valid_y2": [True] * len(x),
            "valid_co": [True] * len(x) if valid_co is None else valid_co,
        }
    )


@pytest.mark.parametrize(
    "key, x, y1, n_required, expected",
    [
        # 0 and 1.2 are 1 apart although every neighbouring gap is < 1
        ("y1", [0, 1, 2], [0, 0.6, 1.2], 2, False),
        ("y1", [0, 1, 2], [0, 0.6, 1.2], 3, True),
        # replicates spread over the threshold are averaged and count once
        ("y1", [0, 0, 0], [0, 1, 2], 1, False),
        ("y1", [0, 0, 0], [0, 1, 2], 2, True),
        # missing output values are ignored
        ("y1", [0, 1, 2], [0, float("nan"), 5], 3, True),
        # inputs are packed directly
        ("x", [0, 0.6, 1.2], [0, 0, 0], 2, False),
        ("x", [0, 0.6, 1.2], [0, 0, 0], 3, True),
        ("x", [1, 1, 1], [0, 5, 10], 2, True),
    ],
)
def test_SufficientDistinguishableValues(key, x, y1, n_required, expected):
    condition = _sufficient_condition(key, 1.0, n_required)
    assert (
        condition.evaluate(None, _sufficient_domain(), _sufficient_experiments(x, y1))
        is expected
    )


def test_SufficientDistinguishableValues_any_column_short():
    experiments = _sufficient_experiments([0, 1, 2], [0, 5, 10], y2=[0, 0, 0])
    condition = data_models.SufficientDistinguishableValues(
        inputs=[
            data_models.DistinguishableParameters(
                key="x", threshold=1.0, n_different_elements=3
            )
        ],
        outputs=[
            data_models.DistinguishableParameters(
                key="y1", threshold=1.0, n_different_elements=3
            ),
            data_models.DistinguishableParameters(
                key="y2", threshold=1.0, n_different_elements=2
            ),
        ],
    )
    assert condition.evaluate(None, _sufficient_domain(), experiments) is True
    condition.outputs[1].n_different_elements = 1
    assert condition.evaluate(None, _sufficient_domain(), experiments) is False
    assert condition.evaluate(None, _sufficient_domain(), None) is True


@pytest.mark.parametrize(
    "key, c, co, valid_co, n_required, expected",
    [
        # categories count once each, the threshold is ignored
        ("c", ["a", "b", "a", "b"], None, None, 2, False),
        ("c", ["a", "b", "a", "b"], None, None, 3, True),
        # only valid output values are used
        ("co", None, ["u", "v", "w", "v"], None, 3, False),
        ("co", None, ["u", "v", "w", "v"], [True, True, False, True], 3, True),
    ],
)
def test_SufficientDistinguishableValues_categorical(
    key, c, co, valid_co, n_required, expected
):
    condition = _sufficient_condition(key, 100.0, n_required)
    experiments = _sufficient_experiments(
        [0, 1, 2, 3], [0, 1, 2, 3], c=c, co=co, valid_co=valid_co
    )
    assert condition.evaluate(None, _sufficient_domain(), experiments) is expected
