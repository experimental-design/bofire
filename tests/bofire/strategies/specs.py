# TODO: remove this and use tests.bofire.data_models.specs instead

import uuid
from typing import List

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MinimizeObjective


def get_invalids(valid: dict) -> List[dict]:
    return [{k: v for k, v in valid.items() if k != k_} for k_ in valid if k_ != "type"]


INVALID_SPECS = [
    [1, 2, 3],
    {"a": "qwe"},
]


objective = MinimizeObjective(w=1)

VALID_CONTINUOUS_INPUT_FEATURE_SPEC = {
    "type": "ContinuousInput",
    "key": str(uuid.uuid4()),
    "bounds": (3, 5.3),
}

VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC = {
    "type": "ContinuousInput",
    "key": str(uuid.uuid4()),
    "bounds": (3, 3),
}

VALID_DISCRETE_INPUT_FEATURE_SPEC = {
    "type": "DiscreteInput",
    "key": str(uuid.uuid4()),
    "values": [1.0, 2.0],
}

VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC = {
    "type": "DiscreteInput",
    "key": str(uuid.uuid4()),
    "values": [2.0],
}

VALID_CONTINUOUS_DESCRIPTOR_INPUT_FEATURE_SPEC = {
    "type": "ContinuousDescriptorInput",
    "key": str(uuid.uuid4()),
    "bounds": (3, 5.3),
    "descriptors": ["d1", "d2"],
    "values": [1.0, 2.0],
}

VALID_CATEGORICAL_INPUT_FEATURE_SPEC = {
    "type": "CategoricalInput",
    "key": str(uuid.uuid4()),
    "categories": ["c1", "c2", "c3"],
    # "allowed": [True, True, False],
}

VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC = {
    "type": "CategoricalDescriptorInput",
    "key": str(uuid.uuid4()),
    "categories": ["c1", "c2", "c3"],
    # "allowed": [True, True, False],
    "descriptors": ["d1", "d2"],
    "values": [
        [1, 2],
        [3, 7],
        [5, 1],
    ],
}


VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC = {
    "type": "CategoricalDescriptorInput",
    "key": str(uuid.uuid4()),
    "categories": ["c1", "c2", "c3"],
    "allowed": [False, True, True],
    "descriptors": ["d1", "d2"],
    "values": [
        [1, 2],
        [3, 7],
        [3, 1],
    ],
}

VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC = {
    "type": "CategoricalInput",
    "key": str(uuid.uuid4()),
    "categories": ["c1", "c2", "c3"],
    "allowed": [True, False, False],
}

VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC = {
    "type": "CategoricalDescriptorInput",
    "key": str(uuid.uuid4()),
    "categories": ["c1", "c2", "c3"],
    "allowed": [True, False, False],
    "descriptors": ["d1", "d2"],
    "values": [
        [1, 2],
        [3, 7],
        [5, 1],
    ],
}

VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC = {
    "type": "ContinuousOutput",
    "key": str(uuid.uuid4()),
}

FEATURE_SPECS = {
    ContinuousInput: {
        "valids": [
            VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
            VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
        ],
    },
    DiscreteInput: {
        "valids": [
            VALID_DISCRETE_INPUT_FEATURE_SPEC,
            VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_DISCRETE_INPUT_FEATURE_SPEC),
            *[
                {
                    **VALID_DISCRETE_INPUT_FEATURE_SPEC,
                    "values": values,
                }
                for values in [[], [1.0, 1.0], [1.0, "a"]]
            ],
        ],
    },
    ContinuousDescriptorInput: {
        "valids": [VALID_CONTINUOUS_DESCRIPTOR_INPUT_FEATURE_SPEC],
        "invalids": [
            *get_invalids(VALID_CONTINUOUS_DESCRIPTOR_INPUT_FEATURE_SPEC),
            *[
                {
                    **VALID_CONTINUOUS_DESCRIPTOR_INPUT_FEATURE_SPEC,
                    "descriptors": descriptors,
                    "values": values,
                }
                for descriptors, values in [
                    ([], []),
                    (["a", "b"], [1]),
                    (["a", "b"], [1, 2, 3]),
                ]
            ],
        ],
    },
    CategoricalInput: {
        "valids": [
            VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
            {
                **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
                "allowed": [True, False, True],
            },
            VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            *[
                {
                    **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
                    "categories": categories,
                    "allowed": allowed,
                }
                for categories, allowed in [
                    ([], []),
                    (["1"], [False]),
                    (["1", "2"], [False, False]),
                    (["1", "1"], None),
                    (["1", "1", "2"], None),
                    (["1", "2"], [True]),
                    (["1", "2"], [True, False, True]),
                    (["1"], []),
                    (["1"], [True]),
                ]
            ],
        ],
    },
    CategoricalDescriptorInput: {
        "valids": [
            VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
            {
                **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
                "allowed": [True, False, True],
            },
            VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC),
            *[
                {
                    **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
                    "categories": categories,
                    "descriptors": descriptors,
                    "values": values,
                }
                for categories, descriptors, values in [
                    (["c1", "c2"], ["d1", "d2", "d3"], []),
                    (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3]]),
                    (
                        ["c1", "c2"],
                        ["d1", "d2", "d3"],
                        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    ),
                    (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [1, 2]]),
                    (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [1, 2, 3, 4]]),
                    (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [1, 2, 3]]),
                ]
            ],
        ],
    },
    ContinuousOutput: {
        "valids": [
            VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
            {
                **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
                "objective": objective,
            },
            {
                **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
                "objective": None,
            },
        ],
        "invalids": [*get_invalids(VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC)],
    },
}


VALID_NCHOOSEKE_CONSTRAINT_SPEC = {
    "features": ["f1", "f2", "f3"],
    "min_count": 1,
    "max_count": 1,
    "none_also_valid": False,
}

VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "LinearEqualityConstraint",
    "features": ["f1", "f2", "f3"],
    "coefficients": [1, 2, 3],
    "rhs": 1.5,
}

VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "LinearInequalityConstraint",
    "features": ["f1", "f2", "f3"],
    "coefficients": [1, 2, 3],
    "rhs": 1.5,
}

VALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearEqualityConstraint",
    "features": ["f1", "f2"],
    "expression": "f1*f2",
}

VALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearInequalityConstraint",
    "features": ["f1", "f2"],
    "expression": "f1*f2",
}

INVALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearEqualityConstraint",
    "expression": [5, 7, 8],
}

INVALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC = {
    "type": "NonlinearInequalityConstraint",
    "expression": [5, 7, 8],
}

VALID_LINEAR_EQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        (["f1", "f2"], [-0.4, 1.4]),
    ]
]

VALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        (["f1", "f2"], [-0.4, 1.4]),
    ]
]

INVALID_LINEAR_EQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        ([], []),
        ([], [1]),
        (["f1", "f2"], [-0.4]),
        (["f1", "f2"], [-0.4, 1.4, 4.3]),
        (["f1", "f1"], [1, 1]),
        (["f1", "f1", "f2"], [1, 1, 1]),
    ]
]

INVALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS = [
    {
        **VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
        "features": features,
        "coefficients": coefficients,
    }
    for features, coefficients in [
        ([], []),
        ([], [1]),
        (["f1", "f2"], [-0.4]),
        (["f1", "f2"], [-0.4, 1.4, 4.3]),
        (["f1", "f1"], [1, 1]),
        (["f1", "f1", "f2"], [1, 1, 1]),
    ]
]


CONSTRAINT_SPECS = {
    NChooseKConstraint: {
        "valids": [VALID_NCHOOSEKE_CONSTRAINT_SPEC],
        "invalids": INVALID_SPECS + get_invalids(VALID_NCHOOSEKE_CONSTRAINT_SPEC),
    },
    LinearEqualityConstraint: {
        "valids": [VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC]
        + VALID_LINEAR_EQUALITY_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC)
        + INVALID_LINEAR_EQUALITY_CONSTRAINT_SPECS,
    },
    LinearInequalityConstraint: {
        "valids": [VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC]
        + VALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS,
        "invalids": INVALID_SPECS
        + get_invalids(VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC)
        + INVALID_LINEAR_INEQUALITY_CONSTRAINT_SPECS,
    },
    NonlinearEqualityConstraint: {
        "valids": [VALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC],
        "invalids": [INVALID_NONLINEAR_EQUALITY_CONSTRAINT_SPEC],
    },
    NonlinearInequalityConstraint: {
        "valids": [VALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC],
        "invalids": [INVALID_NONLINEAR_INEQUALITY_CONSTRAINT_SPEC],
    },
}
