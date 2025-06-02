import random

import bofire.data_models.constraints.api as constraints
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    constraints.ProductEqualityConstraint,
    lambda: {
        "features": ["f1", "f2", "f3"],
        "exponents": [random.randint(1, 10) for _ in range(3)],
        "rhs": random.random(),
        "sign": 1,
    },
)

specs.add_valid(
    constraints.ProductInequalityConstraint,
    lambda: {
        "features": ["f1", "f2", "f3"],
        "exponents": [random.randint(1, 10) for _ in range(3)],
        "rhs": random.random(),
        "sign": 1,
    },
)

specs.add_valid(
    constraints.LinearEqualityConstraint,
    lambda: {
        "features": ["f1", "f2", "f3"],
        "coefficients": [random.randint(1, 10) for _ in range(3)],
        "rhs": random.random(),
    },
)
specs.add_valid(
    constraints.LinearInequalityConstraint,
    lambda: {
        "features": ["f1", "f2", "f3"],
        "coefficients": [random.randint(1, 10) for _ in range(3)],
        "rhs": random.random(),
    },
)
specs.add_valid(
    constraints.NonlinearEqualityConstraint,
    lambda: {
        "expression": "f1*f2",
        "jacobian_expression": "[f2,f1,0]",
        "hessian_expression": "[[0,1,0],[1,0,0],[0,0,0]]",
        "features": ["f1", "f2", "f3"],
    },
)
specs.add_valid(
    constraints.NonlinearInequalityConstraint,
    lambda: {
        "expression": "f1*f2",
        "jacobian_expression": "[f2,f1,0]",
        "hessian_expression": "[[0,1,0],[1,0,0],[0,0,0]]",
        "features": ["f1", "f2", "f3"],
    },
)
specs.add_valid(
    constraints.NChooseKConstraint,
    lambda: {
        "features": ["f1", "f2", "f3"],
        "min_count": 1,
        "max_count": 1,
        "none_also_valid": False,
    },
)

specs.add_valid(
    constraints.InterpointEqualityConstraint,
    lambda: {
        "features": ["f1"],
        "multiplicity": 3,
    },
)

specs.add_invalid(
    constraints.InterpointEqualityConstraint,
    lambda: {
        "features": ["f1"],
        "multiplicity": 1,
    },
    error=ValueError,
)

specs.add_valid(
    constraints.CategoricalExcludeConstraint,
    lambda: {
        "features": ["solvent", "catalyst"],
        "logical_op": "AND",
        "conditions": [
            constraints.SelectionCondition(
                selection=["Acetone", "THF"],
            ).model_dump(),
            constraints.SelectionCondition(
                selection=["alpha", "beta"],
            ).model_dump(),
        ],
    },
)

specs.add_valid(
    constraints.CategoricalExcludeConstraint,
    lambda: {
        "features": ["solvent", "temperature"],
        "logical_op": "AND",
        "conditions": [
            constraints.SelectionCondition(
                selection=["Acetone", "THF"],
            ).model_dump(),
            constraints.ThresholdCondition(
                threshold=50,
                operator=">",
            ).model_dump(),
        ],
    },
)
