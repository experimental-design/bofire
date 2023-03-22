import random

import numpy as np
import pandas as pd
import pytest

import bofire.data_models
import tests.bofire.data_models.specs.api as specs
from bofire.data_models.api import Domain, Inputs
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.strategies.doe.utils import get_formula_from_string
from bofire.utils.reduce import reduce_domain
from tests.bofire.data_models.test_samplers import test_PolytopeSampler


def test_get_formula_from_string():
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(domain=domain, model_type="linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(
        domain=domain, model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0**2", "x1**2", "x2**2"]
    model_formula = get_formula_from_string(
        domain=domain, model_type="linear-and-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # fully quadratic
    terms = [
        "1",
        "x0",
        "x1",
        "x2",
        "x0:x1",
        "x0:x2",
        "x1:x2",
        "x0**2",
        "x1**2",
        "x2**2",
    ]
    model_formula = get_formula_from_string(domain=domain, model_type="fully-quadratic")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0**2", "x0:x1"]
    model_formula = get_formula_from_string(
        domain=domain,
        model_type="y ~ 1 + x0 + x0:x1 + {x0**2}",
        rhs_only=False,
    )
    assert all(term in terms_lhs for term in model_formula.terms.lhs)
    assert all(term in model_formula.terms.lhs for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.terms.rhs)
    assert all(term in model_formula.terms.rhs for term in terms_rhs)

    # get formula without model: valid input
    model = "x1 + x2 + x3"
    model = get_formula_from_string(model_type=model)
    assert str(model) == "1 + x1 + x2 + x3"

    # get formula without model: invalid input
    with pytest.raises(AssertionError):
        model = get_formula_from_string("linear")

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model_type=model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert model.terms[i] in terms
        assert terms[i] in model.terms
