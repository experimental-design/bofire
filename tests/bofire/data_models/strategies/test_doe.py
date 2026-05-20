import warnings

import pytest

from bofire.data_models.strategies.doe import DOptimalityCriterion


@pytest.mark.parametrize(
    "formula",
    [
        "1 + x1 + x1**2",
        "1 + x1 + x1*x1",
        "1 + x1 + x1:x1",
    ],
)
def test_doe_optimality_criterion_warns_on_silent_term_drop(formula):
    with pytest.warns(UserWarning, match="Formulaic may silently drop model terms"):
        DOptimalityCriterion(formula=formula)


def test_doe_optimality_criterion_does_not_warn_for_braced_quadratic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DOptimalityCriterion(formula="1 + x1 + {x1**2}")

    assert not any(
        "Formulaic may silently drop model terms" in str(w.message) for w in caught
    )
