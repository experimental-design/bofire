import numpy as np
import pandas as pd
import pytest

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.doe import (
    AOptimalityCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    GOptimalityCriterion,
    KOptimalityCriterion,
)
from bofire.strategies.doe.score import get_optimality_score


@pytest.fixture
def continuous_domain() -> Domain:
    return Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(-1, 1)) for i in (1, 2)],
        outputs=[ContinuousOutput(key="y")],
    )


FACTORIAL = pd.DataFrame({"x1": [-1.0, 1.0, -1.0, 1.0], "x2": [-1.0, -1.0, 1.0, 1.0]})
CLUSTERED = pd.DataFrame({"x1": [0.9, 1.0, 0.9, 1.0], "x2": [0.9, 0.9, 1.0, 1.0]})


def test_d_optimality_score_matches_closed_form(continuous_domain):
    delta = 1e-6
    X = np.column_stack([np.ones(4), FACTORIAL["x1"], FACTORIAL["x2"]])
    expected = -np.linalg.slogdet(X.T @ X + delta * np.eye(3))[1]
    score = get_optimality_score(
        continuous_domain,
        FACTORIAL,
        DOptimalityCriterion(formula="linear", delta=delta),
    )
    assert score == pytest.approx(expected)


def test_g_optimality_score_matches_closed_form(continuous_domain):
    delta = 1e-6
    design = pd.concat([FACTORIAL, pd.DataFrame({"x1": [0.0], "x2": [0.0]})])
    X = np.column_stack([np.ones(5), design["x1"], design["x2"]])
    hat = X @ np.linalg.inv(X.T @ X + delta * np.eye(3)) @ X.T
    score = get_optimality_score(
        continuous_domain,
        design,
        GOptimalityCriterion(formula="linear", delta=delta),
    )
    assert score == pytest.approx(np.max(np.diag(hat)))


@pytest.mark.parametrize(
    "criterion",
    [
        DOptimalityCriterion,
        AOptimalityCriterion,
        EOptimalityCriterion,
        KOptimalityCriterion,
    ],
)
def test_score_is_lower_for_the_better_design(continuous_domain, criterion):
    c = criterion(formula="linear")
    assert get_optimality_score(continuous_domain, FACTORIAL, c) < get_optimality_score(
        continuous_domain, CLUSTERED, c
    )


def test_score_with_categorical_and_discrete_inputs():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x", bounds=(0, 1)),
            CategoricalInput(key="c", categories=["a", "b"]),
            DiscreteInput(key="z", values=[1, 2, 3]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    criterion = DOptimalityCriterion(formula="linear")
    spread = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
            "c": ["a", "a", "b", "b", "a", "b"],
            "z": [1, 3, 3, 1, 2, 2],
        }
    )
    # the categorical never varies: its effect cannot be estimated
    degenerate = spread.assign(c="a")
    assert get_optimality_score(domain, spread, criterion) < get_optimality_score(
        domain, degenerate, criterion
    )


def test_score_ignores_extra_columns(continuous_domain):
    criterion = DOptimalityCriterion(formula="linear")
    with_extra = FACTORIAL.assign(y=[1.0, 2.0, 3.0, 4.0])
    assert get_optimality_score(
        continuous_domain, with_extra, criterion
    ) == get_optimality_score(continuous_domain, FACTORIAL, criterion)


def test_score_raises_on_missing_column(continuous_domain):
    with pytest.raises(ValueError, match="missing columns"):
        get_optimality_score(
            continuous_domain,
            FACTORIAL[["x1"]],
            DOptimalityCriterion(formula="linear"),
        )


def test_score_raises_on_empty_design(continuous_domain):
    with pytest.raises(ValueError, match="at least one experiment"):
        get_optimality_score(
            continuous_domain,
            FACTORIAL.iloc[:0],
            DOptimalityCriterion(formula="linear"),
        )
