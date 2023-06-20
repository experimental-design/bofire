import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as Specs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    ContinuousDescriptorInput,
)

@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (
            Specs.features.valid(ContinuousDescriptorInput).obj(
                key="k",
                bounds=(1, 1),
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            True,
            [1],
        ),
        (
            Specs.features.valid(ContinuousDescriptorInput).obj(
                key="k",
                bounds=(1, 2),
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            False,
            None,
        ),
        (
            Specs.features.valid(ContinuousDescriptorInput).obj(
                key="k",
                bounds=(2, 3),
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            False,
            None,
        ),
    ],
)
def test_continuous_descriptor_input_feature_is_fixed(input_feature, expected, expected_value):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value

@pytest.mark.parametrize(
    "feature, transform_type, values, expected",
    [
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="c", categories=["B", "A", "C"],allowed=[True, True, True]),
            CategoricalEncodingEnum.ORDINAL,
            None,
            (0, 2),
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="c", categories=["B", "A", "C"],allowed=[True, True, True]),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            pd.Series(["A", "B", "C"]),
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="c", categories=["B", "A", "C"], allowed=[True, False, True]
            ),
            CategoricalEncodingEnum.ONE_HOT,
            None,
            ([0, 0, 0], [1, 0, 1]),
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(key="c", categories=["B", "A", "C"]),
            CategoricalEncodingEnum.DUMMY,
            None,
            ([0, 0], [1, 1]),
        ),
    ],
)
def test_categorical_descriptor_get_bounds(feature, transform_type, values, expected):
    lower, upper = feature.get_bounds(transform_type=transform_type, values=values)
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])
    # test the same for the categorical with descriptor
    f = Specs.features.valid(CategoricalDescriptorInput).obj(
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
    c = Specs.features.valid(CategoricalDescriptorInput).obj(
        key="c",
        categories=["B", "A", "C"],
        allowed=[True, True, True],
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
    c1 = Specs.features.valid(CategoricalDescriptorInput).obj(
        key="c",
        categories=["B", "A", "C"],
        descriptors=["d1", "d2"],
        values=[[1, 2], [3, 4], [5, 6]],
    )
    descriptor_values = pd.DataFrame(
        columns=["c_d1", "c_d2", "misc"],
        data=[[1.05, 2.5, 6], [4, 4.5, 9]],
    )
    samples = c1.from_descriptor_encoding(descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series(["B", "A"]))

    c2 = Specs.features.valid(CategoricalDescriptorInput).obj(
        key="c",
        categories=["B", "A", "C"],
        descriptors=["d1", "d2"],
        values=[[1, 2], [3, 4], [5, 6]],
        allowed=[False, True, True],
    )

    samples = c2.from_descriptor_encoding(descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series(["A", "A"]))


def test_categorical_descriptor_to_descriptor_encoding_1d():
    c = Specs.features.valid(CategoricalDescriptorInput).obj(
        key="c",
        categories=["B", "A", "C"],
        allowed=[True, True, True],
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
    "input_feature, expected_with_values, expected",
    [
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="if1",
                categories=["a", "b"],
                allowed=[True, True],
                descriptors=["alpha", "beta"],
                values=[[1, 2], [3, 4]],
            ),
            ([1, 2], [3, 4]),
            ([1, 2], [3, 4]),
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="if2",
                categories=["a", "b", "c"],
                allowed=[True, False, True],
                descriptors=["alpha", "beta"],
                values=[[1, 2], [3, 4], [1, 5]],
            ),
            ([1, 2], [3, 5]),
            ([1, 2], [1, 5]),
        ),
        # (CategoricalInput(key="if2", categories = ["a","b"], allowed = [True, True]), ["a","b"]),
        # (CategoricalInput(key="if3", categories = ["a","b"], allowed = [True, False]), ["a"]),
        # (CategoricalInput(key="if4", categories = ["a","b"], allowed = [True, False]), ["a", "b"]),
        # (ContinuousInput(key="if1", lower_bound=2.5, upper_bound=2.9), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=3.), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=1.), (1,1.)),
    ],
)
def test_categorical_descriptor_feature_get_bounds(
    input_feature, expected_with_values, expected
):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "c"], "if3": ["a", "a"], "if4": ["b", "b"]}
    )
    lower, upper = input_feature.get_bounds(
        transform_type=CategoricalEncodingEnum.DESCRIPTOR,
        values=experiments[input_feature.key],
    )
    assert np.allclose(lower, expected_with_values[0])
    assert np.allclose(upper, expected_with_values[1])
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
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"]
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"]
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series(["c1", "c1"]),
            False,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[False, True, True],
                descriptors=["d1", "d2"],
                values=[
                    [1, 2],
                    [3, 7],
                    [3, 1],
                ],
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
            Specs.features.valid(CategoricalDescriptorInput).obj(),
            pd.Series(["c1", "c4"]),
            True,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(),
            pd.Series(["c1", "c4"]),
            False,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
                descriptors=["d1", "d2"],
                values=[
                    [1, 2],
                    [3, 7],
                    [5, 1],
                ],
            ),
            pd.Series(["c1", "c1"]),
            True,
        ),
        (
            Specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[False, True, True],
                descriptors=["d1", "d2"],
                values=[
                    [1, 2],
                    [3, 7],
                    [3, 1],
                ],
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
def test_categorical_descriptor_input_feature_is_fixed(
    input_feature, expected, expected_value, transform_type
):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value(transform_type) == expected_value


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
    f = Specs.features.valid(CategoricalDescriptorInput).obj(
        key="k", categories=categories, descriptors=descriptors, values=values, 
        allowed=[True for _ in range(len(categories))]
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
    f = Specs.features.valid(ContinuousDescriptorInput).obj(
        key="k",
        bounds=(1, 2),
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
        dict(zip(categories, values)),
        orient="index",
        columns=descriptors,
    )
    f = CategoricalDescriptorInput.from_df("k", df)
    assert f.categories == categories
    assert f.descriptors == descriptors
    assert f.values == values

