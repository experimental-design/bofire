import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalOutput,
)
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective


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
            CategoricalInput(key="if4", categories=["a", "b"], allowed=[False, True]),
            ["b"],
        ),
    ],
)
def test_categorical_input_feature_get_possible_categories(input_feature, expected):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "a"], "if3": ["a", "a"], "if4": ["b", "b"]},
    )
    categories = input_feature.get_possible_categories(experiments[input_feature.key])
    assert categories == expected


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice([1, 2, 3]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
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
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "a"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "b"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([1, 2]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["one", "two", "three"],
                allowed=[True, False, False],
            ),
            pd.Series([1, 2, 3]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
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
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, True, True],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"],
                allowed=[True, False],
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
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"],
                allowed=[True, False],
            ),
            pd.Series(["a", "b"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


def test_cateogorical_input_is_fulfilled():
    feature = CategoricalInput(
        key="a", categories=["B", "A", "C"], allowed=[True, True, False]
    )
    values = pd.Series(["A", "B", "C", "D"], index=[0, 1, 2, 5])
    fulfilled = feature.is_fulfilled(values)
    assert_series_equal(
        fulfilled,
        pd.Series([True, True, False, False], index=[0, 1, 2, 5]),
    )


@pytest.mark.parametrize(
    "key, categories, samples",
    [
        ("c", ["B", "A", "C"], ["A", "A", "C", "B"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"], ["_A_a", "_A_a", "C_c_", "B_b"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            [
                "__c_alpha_c___A_a",
                "__c_alpha_c___A_a",
                "__c_alpha_c__C_c_",
                "__c_alpha_c__B_b",
            ],
        ),
    ],
)
def test_categorical_to_one_hot_encoding(key, categories, samples):
    c = CategoricalInput(key=key, categories=categories)
    samples = pd.Series(samples)
    t_samples = c.to_onehot_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            columns=[f"{key}_{cat_str}" for cat_str in categories],
        ),
    )
    untransformed = c.from_onehot_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "key, categories",
    [
        ("c", ["B", "A", "C"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
        ),
    ],
)
def test_categorical_from_one_hot_encoding(key, categories):
    c = CategoricalInput(key=key, categories=categories)
    one_hot_values = pd.DataFrame(
        columns=[f"{key}_{cat_str}" for cat_str in categories] + ["misc"],
        data=[[0.9, 0.4, 0.2, 6], [0.8, 0.7, 0.9, 9]],
    )
    samples = c.from_onehot_encoding(one_hot_values)
    assert np.all(samples == pd.Series([categories[0], categories[2]]))


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


@pytest.mark.parametrize(
    "key, categories, samples",
    [
        ("c", ["B", "A", "C"], ["A", "A", "C", "B"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"], ["_A_a", "_A_a", "C_c_", "B_b"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            [
                "__c_alpha_c___A_a",
                "__c_alpha_c___A_a",
                "__c_alpha_c__C_c_",
                "__c_alpha_c__B_b",
            ],
        ),
    ],
)
def test_categorical_to_dummy_encoding(key, categories, samples):
    c = CategoricalInput(key=key, categories=categories)
    samples = pd.Series(samples)
    t_samples = c.to_dummy_encoding(samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            columns=[f"{key}_{cat_str}" for cat_str in categories[1:]],
        ),
    )
    untransformed = c.from_dummy_encoding(t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "key, categories",
    [
        ("c", ["B", "A", "C"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
        ),
    ],
)
def test_categorical_from_dummy_encoding(key, categories):
    c = CategoricalInput(key=key, categories=categories)
    one_hot_values = pd.DataFrame(
        columns=[f"{key}_{cat_str}" for cat_str in categories[1:]] + ["misc"],
        data=[[0.9, 0.05, 6], [0.1, 0.1, 9]],
    )
    samples = c.from_dummy_encoding(one_hot_values)
    assert np.all(samples == pd.Series([categories[1], categories[0]]))


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
                key="c",
                categories=["B", "A", "C"],
                allowed=[True, False, True],
            ),
            CategoricalEncodingEnum.ONE_HOT,
            pd.Series(["A", "B", "C"]),
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            CategoricalInput(
                key="c",
                categories=["B", "A", "C"],
                allowed=[True, False, True],
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
            (True, [1, 2], CategoricalEncodingEnum.DESCRIPTOR),
        ]
    ],
)
def test_categorical_input_feature_is_fixed(
    input_feature,
    expected,
    expected_value,
    transform_type,
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


def test_categorical_output_call():
    test_df = pd.DataFrame(data=[[0.7, 0.3], [0.2, 0.8]], columns=["c1", "c2"])
    categorical_output = CategoricalOutput(
        key="a",
        categories=["c1", "c2"],
        objective=ConstrainedCategoricalObjective(
            categories=["c1", "c2"],
            desirability=[True, False],
        ),
    )
    output = categorical_output(test_df, test_df)
    assert output.tolist() == test_df["c1"].tolist()
