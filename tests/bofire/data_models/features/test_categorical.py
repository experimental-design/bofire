import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


import tests.bofire.data_models.specs.api as Specs
from bofire.data_models.domain.api import  Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
)
from bofire.data_models.surrogates.api import ScalerEnum


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            Specs.features.valid(CategoricalInput).obj(key="if1", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            Specs.features.valid(CategoricalInput).obj(key="if2", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            Specs.features.valid(CategoricalInput).obj(key="if3", categories=["a", "b"], allowed=[True, False]),
            ["a"],
        ),
        (
            Specs.features.valid(CategoricalInput).obj(key="if4", categories=["a", "b"], allowed=[False, True]),
            ["b"],
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
            Specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "a"]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series(["c1", "c1"]),
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
            Specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            True,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            False,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "a"]),
            True,
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "b"]),
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
            Specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, True, True],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"], allowed=[True, False]
            ),
            pd.Series(["a", "a"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)

@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            Specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"], allowed=[True, False]
            ),
            pd.Series(["a", "b"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


@pytest.mark.parametrize("key", ["c", "c_alpha"])
def test_categorical_to_one_hot_encoding(key):
    c = Specs.features.valid(CategoricalInput).obj(key=key, categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_onehot_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            columns=[f"{key}_B", f"{key}_A", f"{key}_C"],
        ),
    )
    untransformed = c.from_onehot_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize("key", ["c", "c_alpha"])
def test_categorical_from_one_hot_encoding(key):
    c = Specs.features.valid(CategoricalInput).obj(key=key, categories=["B", "A", "C"])
    one_hot_values = pd.DataFrame(
        columns=[f"{key}_B", f"{key}_A", f"{key}_C", "misc"],
        data=[[0.9, 0.4, 0.2, 6], [0.8, 0.7, 0.9, 9]],
    )
    samples = c.from_onehot_encoding(one_hot_values)
    assert np.all(samples == pd.Series(["B", "C"]))


def test_categorical_from_one_hot_encoding_invalid():
    c = Specs.features.valid(CategoricalInput).obj(key="c", categories=["B", "A", "C"])
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


@pytest.mark.parametrize("key", ["c", "c_alpha"])
def test_categorical_to_dummy_encoding(key):
    c = Specs.features.valid(CategoricalInput).obj(key=key, categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_dummy_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            columns=[f"{key}_A", f"{key}_C"],
        ),
    )
    untransformed = c.from_dummy_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize("key", ["c", "c_alpha"])
def test_categorical_from_dummy_encoding(key):
    c = Specs.features.valid(CategoricalInput).obj(key=key, categories=["B", "A", "C"])
    one_hot_values = pd.DataFrame(
        columns=[f"{key}_A", f"{key}_C", "misc"],
        data=[[0.9, 0.05, 6], [0.1, 0.1, 9]],
    )
    samples = c.from_dummy_encoding(one_hot_values)
    assert np.all(samples == pd.Series(["A", "B"]))


def test_categorical_to_label_encoding():
    c = Specs.features.valid(CategoricalInput).obj(key="c", categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_ordinal_encoding(samples)
    assert_series_equal(t_samples, pd.Series([1, 1, 2, 0], name="c"))
    untransformed = c.from_ordinal_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "feature, transform_type, values, expected",
    [
        (
            Specs.features.valid(CategoricalInput).obj(
                key="c", categories=["B", "A", "C"],allowed=[True, True, True]),
            CategoricalEncodingEnum.ORDINAL,
            None,
            (0, 2),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                key="c", categories=["B", "A", "C"],allowed=[True, True, True]),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            pd.Series(["A", "B", "C"]),
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 0, 1]),
        ),
        (
            Specs.features.valid(CategoricalInput).obj(key="c", categories=["B", "A", "C"]),
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

@pytest.mark.parametrize(
    "input_feature, expected, expected_value, transform_type",
    [
        (
            Specs.features.valid(CategoricalInput).obj(key="k", categories=categories, allowed=allowed),
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
            Specs.features.valid(CategoricalInput).obj(key="k", categories=categories, allowed=allowed),
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
            Specs.features.valid(CategoricalInput).obj(key="k", categories=categories, allowed=allowed),
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
    "specs",
    [
        ({"x4": CategoricalEncodingEnum.ONE_HOT}),
        ({"x1": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": ScalerEnum.NORMALIZE}),
        ({"x2": CategoricalEncodingEnum.DESCRIPTOR}),
    ],
)
def test_inputs_validate_transform_specs_invalid(specs):
    inps = Inputs(
        features=[
            Specs.features.valid(CategoricalInput).obj(key="x2", categories=["apple", "banana"]
                ,allowed=[True,True]),
           
        ]
    )
    with pytest.raises(ValueError):
        inps._validate_transform_specs(specs)


def test_categorical_output():
    feature = Specs.features.valid(CategoricalOutput).obj(
        key="a", categories=["alpha", "beta", "gamma"], objective=[1.0, 0.0, 0.1]
    )

    assert feature.to_dict() == {"alpha": 1.0, "beta": 0.0, "gamma": 0.1}
    data = pd.Series(data=["alpha", "beta", "beta", "gamma"], name="a")
    assert_series_equal(feature(data), pd.Series(data=[1.0, 0.0, 0.0, 0.1], name="a"))
