import random
import uuid

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic.error_wrappers import ValidationError

from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Features,
    InputFeature,
    InputFeatures,
    OutputFeature,
    OutputFeatures,
)
from bofire.domain.objectives import (
    MaximizeSigmoidObjective,
    MinimizeObjective,
    Objective,
)
from bofire.utils.enum import CategoricalEncodingEnum, ScalerEnum
from tests.bofire.domain.utils import get_invalids

objective = MinimizeObjective(w=1)

VALID_CONTINUOUS_INPUT_FEATURE_SPEC = {
    "type": "ContinuousInput",
    "key": str(uuid.uuid4()),
    "lower_bound": 3,
    "upper_bound": 5.3,
}

VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC = {
    "type": "ContinuousInput",
    "key": str(uuid.uuid4()),
    "lower_bound": 3.0,
    "upper_bound": 3.0,
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
    "lower_bound": 3,
    "upper_bound": 5.3,
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


@pytest.mark.parametrize(
    "cls, spec, n",
    [
        (cls, valid, n)
        for cls, data in FEATURE_SPECS.items()
        for valid in data["valids"]
        for n in [1, 5]
    ],
)
def test_input_feature_sample(cls, spec, n):
    feature = cls(**spec)
    if isinstance(feature, InputFeature):
        samples = feature.sample(n)
        feature.validate_candidental(samples)


@pytest.mark.parametrize(
    "cls, spec",
    [(cls, valid) for cls, data in FEATURE_SPECS.items() for valid in data["valids"]],
)
def test_valid_feature_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)
    assert isinstance(res.__str__(), str)


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, valid)
        for cls, data in FEATURE_SPECS.items()
        for valid in data["valids"]
        if cls != ContinuousOutput
    ],
)
def test_sample(cls, spec):
    feat = cls(**spec)
    samples = feat.sample(n=100)
    feat.validate_candidental(samples)


@pytest.mark.parametrize(
    "cls, spec",
    [(cls, valid) for cls, data in FEATURE_SPECS.items() for valid in data["valids"]],
)
def test_feature_serialize(cls, spec):
    res = cls(**spec)
    res2 = Feature.from_dict(res.dict())
    assert res == res2


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, invalid)
        for cls, data in FEATURE_SPECS.items()
        for invalid in data["invalids"]
    ],
)
def test_invalid_feature_specs(cls, spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        _ = cls(**spec)


@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (ContinuousInput(key="k", lower_bound=1, upper_bound=1), True, [1]),
        (ContinuousInput(key="k", lower_bound=1, upper_bound=2), False, None),
        (ContinuousInput(key="k", lower_bound=2, upper_bound=3), False, None),
        (
            ContinuousDescriptorInput(
                key="k",
                lower_bound=1,
                upper_bound=1,
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            True,
            [1],
        ),
        (
            ContinuousDescriptorInput(
                key="k",
                lower_bound=1,
                upper_bound=2,
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            False,
            None,
        ),
        (
            ContinuousDescriptorInput(
                key="k",
                lower_bound=2,
                upper_bound=3,
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            False,
            None,
        ),
    ],
)
def test_continuous_input_feature_is_fixed(input_feature, expected, expected_value):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            ContinuousInput(key="if1", lower_bound=0.5, upper_bound=4.0),
            (0.5, 4.0),
        ),
        (ContinuousInput(key="if1", lower_bound=2.5, upper_bound=2.9), (1, 3.0)),
        (ContinuousInput(key="if2", lower_bound=1.0, upper_bound=3.0), (1, 3.0)),
        (ContinuousInput(key="if2", lower_bound=1.0, upper_bound=1.0), (1, 1.0)),
    ],
)
def test_continuous_input_feature_get_bounds(input_feature, expected):
    experiments = pd.DataFrame({"if1": [1.0, 2.0, 3.0], "if2": [1.0, 1.0, 1.0]})
    lower, upper = input_feature.get_bounds(values=experiments[input_feature.key])
    assert (lower[0], upper[0]) == expected
    lower, upper = input_feature.get_bounds()
    assert (lower[0], upper[0]) == (
        input_feature.lower_bound,
        input_feature.upper_bound,
    )


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.0, 3.0, 3.0]),
            False,
        ),
    ],
)
def test_continuous_input_feature_validate_valid(input_feature, values, strict):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.0, "mama"]),
            True,
        ),
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.0, "mama"]),
            False,
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.0, 3.0, 3.0]),
            True,
        ),
    ],
)
def test_continuous_input_feature_validate_invalid(input_feature, values, strict):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.0, 3.0, 3.0]),
        ),
    ],
)
def test_continuous_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.1, "a"]),
        ),
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([2.9, 4.0]),
        ),
        (
            ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([4.0, 6]),
        ),
        (
            ContinuousInput(**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC),
            pd.Series([3.1, 3.2, 3.4]),
        ),
    ],
)
def test_continuous_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "feature, xt, expected",
    [
        (
            ContinuousInput(key="a", lower_bound=0, upper_bound=10),
            pd.Series(np.linspace(0, 1, 11)),
            np.linspace(0, 10, 11),
        ),
        (
            ContinuousInput(key="a", lower_bound=-10, upper_bound=20),
            pd.Series(np.linspace(0, 1)),
            np.linspace(-10, 20),
        ),
    ],
)
def test_continuous_input_feature_from_unit_range(feature, xt, expected):
    x = feature.from_unit_range(xt)
    assert np.allclose(x.values, expected)


@pytest.mark.parametrize(
    "feature, x, expected, real",
    [
        (
            ContinuousInput(key="a", lower_bound=0, upper_bound=10),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            ContinuousInput(key="a", lower_bound=-10, upper_bound=20),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            True,
        ),
        (
            ContinuousInput(key="a", lower_bound=0, upper_bound=10),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            False,
        ),
        (
            ContinuousInput(key="a", lower_bound=-10, upper_bound=20),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            False,
        ),
        (
            ContinuousInput(key="a", lower_bound=0, upper_bound=9),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            ContinuousInput(key="a", lower_bound=0, upper_bound=9),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 10 / 9, 11),
            False,
        ),
    ],
)
def test_continuous_input_feature_to_unit_range(feature, x, expected, real):
    xt = feature.to_unit_range(x)
    assert np.allclose(xt.values, expected, real)


@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (DiscreteInput(**VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC), True, [2.0]),
        (DiscreteInput(**VALID_DISCRETE_INPUT_FEATURE_SPEC), False, None),
    ],
)
def test_discrete_input_feature_is_fixed(input_feature, expected, expected_value):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value


@pytest.mark.parametrize(
    "input_feature, expected_lower, expected_upper",
    [
        (
            DiscreteInput(**VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC),
            VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC["values"][0],
            VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC["values"][0],
        ),
        (
            DiscreteInput(**VALID_DISCRETE_INPUT_FEATURE_SPEC),
            VALID_DISCRETE_INPUT_FEATURE_SPEC["values"][0],
            VALID_DISCRETE_INPUT_FEATURE_SPEC["values"][1],
        ),
    ],
)
def test_discrete_input_feature_bounds(input_feature, expected_lower, expected_upper):
    assert input_feature.upper_bound == expected_upper
    assert input_feature.lower_bound == expected_lower


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            DiscreteInput(key="if1", values=[2.0, 3.0]),
            (1.0, 4.0),
        ),
        (
            DiscreteInput(key="if1", values=[0.0, 3.0]),
            (0.0, 4.0),
        ),
        (
            DiscreteInput(key="if1", values=[2.0, 5.0]),
            (1.0, 5.0),
        ),
    ],
)
def test_discrete_input_feature_get_bounds(input_feature, expected):
    experiments = pd.DataFrame(
        {"if1": [1.0, 2.0, 3.0, 4.0], "if2": [1.0, 1.0, 1.0, 1.0]}
    )
    lower, upper = input_feature.get_bounds(values=experiments[input_feature.key])
    assert (lower[0], upper[0]) == expected
    lower, upper = input_feature.get_bounds()
    assert (lower[0], upper[0]) == (
        input_feature.lower_bound,
        input_feature.upper_bound,
    )


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            DiscreteInput(**VALID_DISCRETE_INPUT_FEATURE_SPEC),
            pd.Series([random.choice([1.0, 2.0]) for _ in range(20)]),
        ),
        (
            DiscreteInput(**VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC),
            pd.Series([2.0, 2.0, 2.0]),
        ),
    ],
)
def test_discrete_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            DiscreteInput(**VALID_DISCRETE_INPUT_FEATURE_SPEC),
            pd.Series([random.choice([1.0, 3.0, 2.0]) for _ in range(20)]),
        ),
        (
            DiscreteInput(**VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC),
            pd.Series([1.0, 2.0, 2.0]),
        ),
    ],
)
def test_discrete_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="if1", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            CategoricalInput(key="if2", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            CategoricalInput(key="if3", categories=["a", "b"], allowed=[True, False]),
            ["a"],
        ),
        (
            CategoricalInput(key="if4", categories=["a", "b"], allowed=[True, False]),
            ["a", "b"],
        ),
    ],
)
def test_categorical_input_feature_get_possible_categories(input_feature, expected):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "a"], "if3": ["a", "a"], "if4": ["b", "b"]}
    )
    categories = input_feature.get_possible_categories(experiments[input_feature.key])
    assert categories == expected


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c1"]),
            False,
        ),
    ],
)
def test_categorical_input_feature_validate_valid(input_feature, values, strict):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c4"]),
            True,
        ),
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c4"]),
            False,
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c1"]),
            True,
        ),
        (
            CategoricalInput(
                key="a", categories=["c1", "c2", "c3"], allowed=[True, True, False]
            ),
            pd.Series(["c1", "c2"]),
            True,
        ),
    ],
)
def test_categorical_input_feature_validate_invalid(input_feature, values, strict):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c1"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c2", "c4"]),
        ),
        (
            CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            pd.Series(["c1", "c2"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


def test_categorical_to_one_hot_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_onehot_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            columns=["c_B", "c_A", "c_C"],
        ),
    )
    untransformed = c.from_onehot_encoding(t_samples)
    assert np.all(samples == untransformed)


def test_categorical_from_one_hot_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    one_hot_values = pd.DataFrame(
        columns=["c_B", "c_A", "c_C", "misc"],
        data=[[0.9, 0.4, 0.2, 6], [0.8, 0.7, 0.9, 9]],
    )
    samples = c.from_onehot_encoding(one_hot_values)
    assert np.all(samples == pd.Series(["B", "C"]))


def test_categorical_from_one_hot_encoding_invalid():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    one_hot_values = pd.DataFrame(
        columns=["c_B", "c_A", "misc"],
        data=[
            [
                0.9,
                0.4,
                0.2,
            ],
            [0.8, 0.7, 0.9],
        ],
    )
    with pytest.raises(ValueError):
        c.from_onehot_encoding(one_hot_values)


def test_categorical_to_dummy_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_dummy_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            columns=["c_A", "c_C"],
        ),
    )
    untransformed = c.from_dummy_encoding(t_samples)
    assert np.all(samples == untransformed)


def test_categorical_from_dummy_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    one_hot_values = pd.DataFrame(
        columns=["c_A", "c_C", "misc"],
        data=[[0.9, 0.05, 6], [0.1, 0.1, 9]],
    )
    samples = c.from_dummy_encoding(one_hot_values)
    assert np.all(samples == pd.Series(["A", "B"]))


def test_categorical_to_label_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_ordinal_encoding(samples)
    assert_series_equal(t_samples, pd.Series([1, 1, 2, 0], name="c"))
    untransformed = c.from_ordinal_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "feature, transform_type, values, expected",
    [
        (
            CategoricalInput(key="c", categories=["B", "A", "C"]),
            CategoricalEncodingEnum.ORDINAL,
            None,
            (0, 2),
        ),
        (
            CategoricalInput(key="c", categories=["B", "A", "C"]),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            CategoricalInput(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            pd.Series(["A", "B", "C"]),
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            CategoricalInput(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 0, 1]),
        ),
        (
            CategoricalInput(key="c", categories=["B", "A", "C"]),
            CategoricalEncodingEnum.DUMMY,
            None,
            ([0, 0], [1, 1]),
        ),
    ],
)
def test_categorical_get_bounds(feature, transform_type, values, expected):
    lower, upper = feature.get_bounds(transform_type=transform_type, values=values)
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])
    # test the same for the categorical with descriptor
    f = CategoricalDescriptorInput(
        key="c",
        categories=feature.categories,
        allowed=feature.allowed,
        descriptors=["alpha", "beta"],
        values=[[1, 2], [3, 4], [5, 6]],
    )
    lower, upper = f.get_bounds(transform_type=transform_type, values=values)
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


def test_categorical_descriptor_to_descriptor_encoding():
    c = CategoricalDescriptorInput(
        key="c",
        categories=["B", "A", "C"],
        descriptors=["d1", "d2"],
        values=[[1, 2], [3, 4], [5, 6]],
    )
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_descriptor_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0]],
            columns=["c_d1", "c_d2"],
        ),
    )
    untransformed = c.from_descriptor_encoding(t_samples)
    assert np.all(samples == untransformed)


def test_categorical_descriptor_from_descriptor_encoding():
    c = CategoricalDescriptorInput(
        key="c",
        categories=["B", "A", "C"],
        descriptors=["d1", "d2"],
        values=[[1, 2], [3, 4], [5, 6]],
    )
    descriptor_values = pd.DataFrame(
        columns=["c_d1", "c_d2", "misc"],
        data=[[1.05, 2.5, 6], [4, 4.5, 9]],
    )
    samples = c.from_descriptor_encoding(descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series(["B", "A"]))


def test_categorical_descriptor_to_descriptor_encoding_1d():
    c = CategoricalDescriptorInput(
        key="c",
        categories=["B", "A", "C"],
        descriptors=["d1"],
        values=[[1], [3], [5]],
    )
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_descriptor_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[3.0], [3.0], [5.0], [1.0]],
            columns=["c_d1"],
        ),
    )
    untransformed = c.from_descriptor_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalDescriptorInput(
                key="if1",
                categories=["a", "b"],
                allowed=[True, True],
                descriptors=["alpha", "beta"],
                values=[[1, 2], [3, 4]],
            ),
            ([1, 2], [3, 4]),
        ),
        (
            CategoricalDescriptorInput(
                key="if2",
                categories=["a", "b", "c"],
                allowed=[True, False, True],
                descriptors=["alpha", "beta"],
                values=[[1, 2], [3, 4], [1, 5]],
            ),
            ([1, 2], [1, 5]),
        ),
        # (CategoricalInputFeature(key="if2", categories = ["a","b"], allowed = [True, True]), ["a","b"]),
        # (CategoricalInputFeature(key="if3", categories = ["a","b"], allowed = [True, False]), ["a"]),
        # (CategoricalInputFeature(key="if4", categories = ["a","b"], allowed = [True, False]), ["a", "b"]),
        # (ContinuousInputFeature(key="if1", lower_bound=2.5, upper_bound=2.9), (1,3.)),
        # (ContinuousInputFeature(key="if2", lower_bound=1., upper_bound=3.), (1,3.)),
        # (ContinuousInputFeature(key="if2", lower_bound=1., upper_bound=1.), (1,1.)),
    ],
)
def test_categorical_descriptor_feature_get_bounds(input_feature, expected):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "c"], "if3": ["a", "a"], "if4": ["b", "b"]}
    )
    lower, upper = input_feature.get_bounds(
        transform_type=CategoricalEncodingEnum.DESCRIPTOR,
        values=experiments[input_feature.key],
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])
    lower, upper = input_feature.get_bounds(
        transform_type=CategoricalEncodingEnum.DESCRIPTOR,
        values=None,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            CategoricalDescriptorInput(
                **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c1", "c1"]),
            False,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c2", "c3"]),
            False,
        ),
    ],
)
def test_categorical_descriptor_input_feature_validate_valid(
    input_feature, values, strict
):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            CategoricalDescriptorInput(
                **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c1", "c4"]),
            True,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c1", "c4"]),
            False,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c1", "c1"]),
            True,
        ),
        (
            CategoricalDescriptorInput(
                **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC
            ),
            pd.Series(["c2", "c3"]),
            True,
        ),
    ],
)
def test_categorical_descriptor_input_feature_validate_invalid(
    input_feature, values, strict
):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, expected, expected_value, transform_type",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
            expected_value,
            transform_type,
        )
        for categories, allowed, expected, expected_value, transform_type in [
            (["1", "2"], None, False, None, None),
            (["1", "2", "3", "4"], [True, False, False, False], True, ["1"], None),
            (["1", "2", "3", "4"], [True, True, False, True], False, None, None),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [0],
                CategoricalEncodingEnum.ORDINAL,
            ),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [1, 0, 0, 0],
                CategoricalEncodingEnum.ONE_HOT,
            ),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [0, 0, 0],
                CategoricalEncodingEnum.DUMMY,
            ),
        ]
    ]
    + [
        (
            CategoricalDescriptorInput(
                key="k",
                categories=["1", "2", "3"],
                allowed=[True, False, False],
                descriptors=["alpha", "beta"],
                values=[[1, 2], [3, 4], [5, 6]],
            ),
            expected,
            expected_value,
            transform_type,
        )
        for expected, expected_value, transform_type in [
            (True, [1, 2], CategoricalEncodingEnum.DESCRIPTOR)
        ]
    ],
)
def test_categorical_input_feature_is_fixed(
    input_feature, expected, expected_value, transform_type
):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value(transform_type) == expected_value


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
        )
        for categories, allowed, expected in [
            (["a", "b", "c"], [True, True, True], ["a", "b", "c"]),
            (["a", "b", "c"], [False, True, True], ["b", "c"]),
        ]
    ],
)
def test_categorical_input_feature_allowed_categories(input_feature, expected):
    assert input_feature.get_allowed_categories() == expected


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
        )
        for categories, allowed, expected in [
            (["a", "b", "c"], [True, True, True], []),
            (["a", "b", "c"], [False, True, True], ["a"]),
        ]
    ],
)
def test_categorical_input_feature_forbidden_categories(input_feature, expected):
    assert input_feature.get_forbidden_categories() == expected


@pytest.mark.parametrize(
    "categories, descriptors, values",
    [
        (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [4, 5, 6]]),
        (
            ["c1", "c2", "c3", "c4"],
            ["d1", "d2", "d3"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
                [4, 5, 6],
            ],
        ),
    ],
)
def test_categorical_descriptor_input_feature_as_dataframe(
    categories, descriptors, values
):
    f = CategoricalDescriptorInput(
        key="k", categories=categories, descriptors=descriptors, values=values
    )
    df = f.to_df()
    assert len(df.columns) == len(descriptors)
    assert len(df) == len(categories)
    assert df.values.tolist() == values


@pytest.mark.parametrize(
    "descriptors, values",
    [
        (["a", "b"], [1.0, 2.0]),
        (["a", "b", "c"], [1.0, 2.0, 3.0]),
    ],
)
def test_continuous_descriptor_input_feature_as_dataframe(descriptors, values):
    f = ContinuousDescriptorInput(
        key="k",
        lower_bound=1.0,
        upper_bound=2.0,
        descriptors=descriptors,
        values=values,
    )
    df = f.to_df()
    assert len(df.columns) == len(descriptors)
    assert len(df) == 1
    assert df.values.tolist()[0] == values


@pytest.mark.parametrize(
    "categories, descriptors, values",
    [
        (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [4, 5, 6]]),
        (
            ["c1", "c2", "c3", "c4"],
            ["d1", "d2", "d3"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
                [4, 5, 6],
            ],
        ),
    ],
)
def test_categorical_descriptor_input_feature_from_dataframe(
    categories, descriptors, values
):
    df = pd.DataFrame.from_dict(
        {category: v for category, v in zip(categories, values)},
        orient="index",
        columns=descriptors,
    )
    f = CategoricalDescriptorInput.from_df("k", df)
    assert f.categories == categories
    assert f.descriptors == descriptors
    assert f.values == values


cont = ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC)
cat = CategoricalInput(**VALID_CATEGORICAL_INPUT_FEATURE_SPEC)
cat_ = CategoricalDescriptorInput(**VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC)
out = ContinuousOutput(**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC)


@pytest.mark.parametrize(
    "unsorted_list, sorted_list",
    [
        (
            [cont, cat_, cat, out],
            [cont, cat_, cat, out],
        ),
        (
            [cont, cat_, cat, out, cat_, out],
            [cont, cat_, cat_, cat, out, out],
        ),
        (
            [cont, out],
            [cont, out],
        ),
        (
            [out, cont],
            [cont, out],
        ),
    ],
)
def test_feature_sorting(unsorted_list, sorted_list):
    assert list(sorted(unsorted_list)) == sorted_list


# test features container
if1 = ContinuousInput(**{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if1"})
if2 = CategoricalInput(**{**VALID_CATEGORICAL_INPUT_FEATURE_SPEC, "key": "if2"})
if3 = ContinuousInput(**{**VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if3"})
if4 = CategoricalInput(**{**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC, "key": "if4"})
if5 = DiscreteInput(**{**VALID_DISCRETE_INPUT_FEATURE_SPEC, "key": "if5"})
if6 = DiscreteInput(**{**VALID_FIXED_DISCRETE_INPUT_FEATURE_SPEC, "key": "if6"})
if7 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if7",
        "allowed": [True, False, True],
    }
)


of1 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of1"})
of2 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of2"})
of3 = ContinuousOutput(key="of3", objective=None)

input_features = InputFeatures(features=[if1, if2])
output_features = OutputFeatures(features=[of1, of2])
features = Features(features=[if1, if2, of1, of2])


@pytest.mark.parametrize("features", [input_features, output_features, features])
def test_features_serialize(features):
    nfeatures = Features(**features.dict())
    assert nfeatures == features


@pytest.mark.parametrize(
    "FeatureContainer, features",
    [
        (Features, ["s"]),
        (Features, [ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC), 5]),
        (InputFeatures, ["s"]),
        (InputFeatures, [ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC), 5]),
        (
            InputFeatures,
            [
                ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
                ContinuousOutput(**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC),
            ],
        ),
        (OutputFeatures, ["s"]),
        (OutputFeatures, [ContinuousOutput(**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC), 5]),
        (
            OutputFeatures,
            [
                ContinuousOutput(**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC),
                ContinuousInput(**VALID_CONTINUOUS_INPUT_FEATURE_SPEC),
            ],
        ),
    ],
)
def test_features_invalid_feature(FeatureContainer, features):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        FeatureContainer(features=features)


@pytest.mark.parametrize(
    "features1, features2, expected_type",
    [
        [input_features, input_features, InputFeatures],
        [output_features, output_features, OutputFeatures],
        [input_features, output_features, Features],
        [output_features, input_features, Features],
        [features, output_features, Features],
        [features, input_features, Features],
        [output_features, features, Features],
        [input_features, features, Features],
    ],
)
def test_features_plus(features1, features2, expected_type):
    returned = features1 + features2
    assert type(returned) == expected_type
    assert len(returned) == (len(features1) + len(features2))


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, [if1, if2, of1, of2]),
        (features, OutputFeature, False, [of1, of2]),
        (input_features, ContinuousInput, False, [if1]),
        (output_features, ContinuousOutput, False, [of1, of2]),
    ],
)
def test_constraints_get(features, FeatureType, exact, expected):
    returned = features.get(FeatureType, exact=exact)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])
    assert type(returned) == type(features)


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, ["if1", "if2", "of1", "of2"]),
        (features, OutputFeature, False, ["of1", "of2"]),
        (input_features, ContinuousInput, False, ["if1"]),
        (output_features, ContinuousOutput, False, ["of1", "of2"]),
    ],
)
def test_features_get_keys(features, FeatureType, exact, expected):
    assert features.get_keys(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "features, key, expected",
    [
        (features, "if1", if1),
        (output_features, "of1", of1),
        (input_features, "if1", if1),
    ],
)
def test_features_get_by_key(features, key, expected):
    returned = features.get_by_key(key)
    assert returned.key == expected.key
    assert id(returned) == id(expected)


@pytest.mark.parametrize(
    "features, key",
    [
        (features, "if133"),
        (output_features, "of3331"),
        (input_features, "if1333333"),
    ],
)
def test_features_get_by_key_invalid(features, key):
    with pytest.raises(KeyError):
        features.get_by_key(key)


@pytest.mark.parametrize(
    "features, expected",
    [
        (InputFeatures(features=[if1, if2]), []),
        (InputFeatures(features=[if1, if2, if3, if4, if5, if6]), [if3, if4, if6]),
    ],
)
def test_input_features_get_fixed(features, expected):
    returned = features.get_fixed()
    assert isinstance(features, InputFeatures)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, expected",
    [
        (InputFeatures(features=[if1, if2]), [if1, if2]),
        (InputFeatures(features=[if1, if2, if3, if4, if5, if6]), [if1, if2, if5]),
    ],
)
def test_input_features_get_free(features, expected):
    returned = features.get_free()
    assert isinstance(features, InputFeatures)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, num_samples, method",
    [
        (features, num_samples, method)
        for features in [
            input_features,
            InputFeatures(features=[if1, if2, if3, if4, if5, if6, if7]),
        ]
        for num_samples in [1, 2, 1024]
        for method in ["UNIFORM", "SOBOL", "LHS"]
    ],
)
def test_input_features_sample(features: InputFeatures, num_samples, method):
    samples = features.sample(num_samples, method=method)
    assert samples.shape == (num_samples, len(features))
    assert list(samples.columns) == features.get_keys()


@pytest.mark.parametrize(
    "specs",
    [
        ({"x4": CategoricalEncodingEnum.ONE_HOT}),
        ({"x1": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": ScalerEnum.NORMALIZE}),
        ({"x2": CategoricalEncodingEnum.DESCRIPTOR}),
    ],
)
def test_input_features_validate_transform_specs_invalid(specs):
    inps = InputFeatures(
        features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=1),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    with pytest.raises(ValueError):
        inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
    ],
)
def test_input_features_validate_transform_valid(specs):
    inps = InputFeatures(
        features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=1),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs, expected_features2idx, expected_features2names",
    [
        (
            {"x2": CategoricalEncodingEnum.ONE_HOT},
            {"x1": (0,), "x2": (2, 3, 4), "x3": (1,)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": ("x3",),
            },
        ),
        (
            {"x2": CategoricalEncodingEnum.DUMMY},
            {"x1": (0,), "x2": (2, 3), "x3": (1,)},
            {"x1": ("x1",), "x2": ("x2_banana", "x2_orange"), "x3": ("x3",)},
        ),
        (
            {"x2": CategoricalEncodingEnum.ORDINAL},
            {"x1": (0,), "x2": (2,), "x3": (1,)},
            {"x1": ("x1",), "x2": ("x2",), "x3": ("x3",)},
        ),
        (
            {"x3": CategoricalEncodingEnum.ONE_HOT},
            {"x1": (0,), "x2": (5,), "x3": (1, 2, 3, 4)},
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": ("x3_apple", "x3_banana", "x3_orange", "x3_cherry"),
            },
        ),
        (
            {"x3": CategoricalEncodingEnum.DESCRIPTOR},
            {"x1": (0,), "x2": (3,), "x3": (1, 2)},
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": (
                    "x3_d1",
                    "x3_d2",
                ),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            },
            {"x1": (0,), "x2": (3, 4, 5), "x3": (1, 2)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": (
                    "x3_d1",
                    "x3_d2",
                ),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            },
            {"x1": (0,), "x2": (5, 6, 7), "x3": (1, 2, 3, 4)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": ("x3_apple", "x3_banana", "x3_orange", "x3_cherry"),
            },
        ),
    ],
)
def test_input_features_get_transform_info(
    specs, expected_features2idx, expected_features2names
):
    inps = InputFeatures(
        features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=1),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
        ]
    )
    features2idx, features2names = inps._get_transform_info(specs)
    assert features2idx == expected_features2idx
    assert features2names == expected_features2names


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": CategoricalEncodingEnum.DUMMY}),
        ({"x2": CategoricalEncodingEnum.ORDINAL}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.DUMMY,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
    ],
)
def test_input_features_transform(specs):
    inps = InputFeatures(
        features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=1),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
        ]
    )
    samples = inps.sample(n=10)
    transformed = inps.transform(experiments=samples, specs=specs)
    untransformed = inps.inverse_transform(experiments=transformed, specs=specs)
    assert_frame_equal(samples, untransformed)


if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = ContinuousInput(
    **{
        **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)

if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)

if4 = CategoricalInput(
    **{
        **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if4",
    }
)

if5 = CategoricalDescriptorInput(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if5",
    }
)

if6 = CategoricalDescriptorInput(
    **{
        **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if6",
    }
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)

input_features1 = InputFeatures(features=[if1, if3, if5])

input_features2 = InputFeatures(
    features=[
        if1,
        if2,
        if3,
        if4,
        if5,
        if6,
    ]
)


@pytest.mark.parametrize(
    "input_features, specs, expected_bounds",
    [
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 1, 0, 0, 0], [5.3, 5, 7, 1, 1, 1]],
        ),
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 1, 0, 0], [5.3, 5, 7, 1, 1]],
        ),
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DUMMY,
            },
            [[3, 0, 0, 0, 0], [5.3, 1, 1, 1, 1]],
        ),
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
            },
            [[3, 0, 0, 0, 0, 0, 0], [5.3, 1, 1, 1, 1, 1, 1]],
        ),
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 1, 0], [5.3, 5, 7, 2]],
        ),
        (
            input_features1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 0, 0], [5.3, 2, 2]],
        ),
        # new domain
        (
            input_features2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3, 3, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0],
                [
                    5.3,
                    3,
                    5,
                    7,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
            ],
        ),
        (
            input_features2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
                "if6": CategoricalEncodingEnum.ONE_HOT,
            },
            [
                [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5.3, 3, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
        ),
        (
            input_features2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [
                    3,
                    3,
                    1,
                    1,
                    1,
                    2,
                    0,
                    0,
                ],
                [5.3, 3, 5, 7, 1, 2, 2, 2],
            ],
        ),
        (
            input_features2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 3, 0, 0, 0, 0], [5.3, 3, 2, 2, 2, 2]],
        ),
        (
            input_features2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3, 3, 0, 1, 2, 0, 0, 0, 0],
                [
                    5.3,
                    3,
                    2,
                    1,
                    2,
                    2,
                    1,
                    0,
                    0,
                ],
            ],
        ),
    ],
)
def test_input_features_get_bounds(input_features, specs, expected_bounds):
    lower, upper = input_features.get_bounds(specs=specs)
    print(lower, upper)
    print(expected_bounds)
    assert np.allclose(
        expected_bounds[0], lower
    )  # torch.equal asserts false due to deviation of 1e-7??
    assert np.allclose(expected_bounds[1], upper)


def test_input_features_get_bounds_fit():
    # at first the fix on the continuous ones is tested
    input_features = InputFeatures(features=[if1, if2])
    experiments = input_features.sample(100)
    experiments["if1"] += [random.uniform(-2, 2) for _ in range(100)]
    experiments["if2"] += [random.uniform(-2, 2) for _ in range(100)]
    opt_bounds = input_features.get_bounds(specs={})
    fit_bounds = input_features.get_bounds(specs={}, experiments=experiments)
    for i, key in enumerate(input_features.get_keys(ContinuousInput)):
        assert fit_bounds[0][i] < opt_bounds[0][i]
        assert fit_bounds[1][i] > opt_bounds[1][i]
        assert fit_bounds[0][i] == experiments[key].min()
        assert fit_bounds[1][i] == experiments[key].max()
    # next test the fix for the CategoricalDescriptor feature
    input_features = InputFeatures(
        features=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
        ]
    )
    experiments = input_features.sample(100)
    experiments["if4"] = [random.choice(if4.categories) for _ in range(100)]
    experiments["if6"] = [random.choice(if6.categories) for _ in range(100)]
    opt_bounds = input_features.get_bounds(
        specs={
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        }
    )
    fit_bounds = input_features.get_bounds(
        {
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        },
        experiments=experiments,
    )
    # check difference in descriptors
    assert opt_bounds[0][-8] == opt_bounds[1][-8] == 1
    assert opt_bounds[0][-7] == opt_bounds[1][-7] == 2
    assert fit_bounds[0][-8] == 1
    assert fit_bounds[0][-7] == 1
    assert fit_bounds[1][-8] == 5
    assert fit_bounds[1][-7] == 7
    # check difference in onehots
    assert opt_bounds[1][-1] == 0
    assert opt_bounds[1][-2] == 0
    assert fit_bounds[1][-1] == 1
    assert fit_bounds[1][-2] == 1


@pytest.mark.parametrize(
    "features, samples",
    [
        (
            output_features,
            pd.DataFrame(
                columns=["of1", "of2"],
                index=range(5),
                data=np.random.uniform(size=(5, 2)),
            ),
        ),
        (
            OutputFeatures(features=[of1, of2, of3]),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
    ],
)
def test_output_features_call(features, samples):
    o = features(samples)
    assert o.shape == (len(samples), len(features.get_keys_by_objective(Objective)))
    assert list(o.columns) == features.get_keys_by_objective(Objective)


@pytest.mark.parametrize(
    "feature, data",
    [
        (
            ContinuousOutput(
                key="of1", objective=MaximizeSigmoidObjective(w=1, tp=15, steepness=0.5)
            ),
            None,
        ),
        (
            ContinuousOutput(
                key="of1", objective=MaximizeSigmoidObjective(w=1, tp=15, steepness=0.5)
            ),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
    ],
)
def test_output_feature_plot(feature, data):
    feature.plot(lower=0, upper=30, experiments=data)
