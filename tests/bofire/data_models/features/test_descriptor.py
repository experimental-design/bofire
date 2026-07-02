import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousDescriptorInput,
)


@pytest.mark.parametrize(
    "key, categories, samples_in, descriptors",
    [
        ("c", ["B", "A", "C"], ["A", "A", "C", "B"], ["d1", "d2"]),
        (
            "c_alpha",
            ["B_b", "_A_a", "C_c_"],
            ["_A_a", "_A_a", "C_c_", "B_b"],
            ["_d1_d", "d2_d_2_"],
        ),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            [
                "__c_alpha_c___A_a",
                "__c_alpha_c___A_a",
                "__c_alpha_c__C_c_",
                "__c_alpha_c__B_b",
            ],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_to_descriptor_encoding(
    key,
    categories,
    samples_in,
    descriptors,
):
    c = CategoricalDescriptorInput(
        key=key,
        categories=categories,
        descriptors=descriptors,
        values=[[1, 2], [3, 4], [5, 6]],
    )
    samples = pd.Series(samples_in)
    t_samples = DescriptorEncoding().encode(c, samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0]],
            columns=[f"{key}_{des_str}" for des_str in descriptors],
        ),
    )
    untransformed = DescriptorEncoding().decode(c, t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "key, categories, descriptors",
    [
        ("c", ["B", "A", "C"], ["d1", "d2"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"], ["_d1_d", "d2_d_2_"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_from_descriptor_encoding(key, categories, descriptors):
    c1 = CategoricalDescriptorInput(
        key=key,
        categories=categories,
        descriptors=descriptors,
        values=[[1, 2], [3, 4], [5, 6]],
    )
    descriptor_values = pd.DataFrame(
        columns=[f"{key}_{des_str}" for des_str in descriptors] + ["misc"],
        data=[[1.05, 2.5, 6], [4, 4.5, 9]],
    )
    samples = DescriptorEncoding().decode(c1, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[0], categories[1]]))

    c2 = CategoricalDescriptorInput(
        key=key,
        categories=categories,
        descriptors=descriptors,
        values=[[1, 2], [3, 4], [5, 6]],
        allowed=[False, True, True],
    )

    samples = DescriptorEncoding().decode(c2, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[1], categories[1]]))


@pytest.mark.parametrize(
    "input_feature, expected_with_values, expected",
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
    input_feature,
    expected_with_values,
    expected,
):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "c"], "if3": ["a", "a"], "if4": ["b", "b"]},
    )
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=experiments[input_feature.key],
    )
    assert np.allclose(lower, expected_with_values[0])
    assert np.allclose(upper, expected_with_values[1])
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=None,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(200)]),
            True,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(200)]),
            False,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series(["c1", "c1"]),
            False,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
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
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice([1, 2, 3]) for _ in range(200)]),
            False,
        ),
    ],
)
def test_categorical_descriptor_input_feature_validate_valid(
    input_feature,
    values,
    strict,
):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalDescriptorInput).obj(),
            pd.Series(["c1", "c4"]),
            True,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(),
            pd.Series(["c1", "c4"]),
            False,
        ),
        (
            specs.features.valid(CategoricalDescriptorInput).obj(
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
            specs.features.valid(CategoricalDescriptorInput).obj(
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
    ],
)
def test_categorical_descriptor_input_feature_validate_invalid(
    input_feature,
    values,
    strict,
):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


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
    categories,
    descriptors,
    values,
):
    f = CategoricalDescriptorInput(
        key="k",
        categories=categories,
        descriptors=descriptors,
        values=values,
    )
    df = f.descriptor_table(f.descriptor_columns(role="descriptor"))
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
        bounds=(1, 2),
        descriptors=descriptors,
        values=values,
    )
    df = f.descriptor_table(f.descriptor_columns(role="descriptor"))
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
    categories,
    descriptors,
    values,
):
    df = pd.DataFrame.from_dict(
        dict(zip(categories, values)),
        orient="index",
        columns=descriptors,
    )
    f = CategoricalDescriptorInput.from_df("k", df)
    assert f.categories == categories
    assert f.descriptor_columns(role="descriptor") == descriptors
    assert f.descriptor_table(descriptors).values.tolist() == values


def test_categorical_descriptor_input_to_pydantic_field():
    feat = CategoricalDescriptorInput(
        key="cat",
        categories=["a", "b"],
        descriptors=["d1", "d2"],
        values=[[1.0, 2.0], [3.0, 4.0]],
    )
    _, field_info = feat.to_pydantic_field()
    assert (
        field_info.description
        == "Categorical, allowed: ['a', 'b'] — descriptors: ['d1', 'd2']"
    )


def test_continuous_descriptor_input_to_pydantic_field():
    from bofire.data_models.features.api import ContinuousDescriptorInput

    feat = ContinuousDescriptorInput(
        key="x",
        bounds=(0, 1),
        descriptors=["d1"],
        values=[0.5],
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is float
    # the deprecated shim is now a plain ContinuousInput; the descriptor table no
    # longer surfaces in the LLM field description.
    assert field_info.description == "Continuous, bounds [0.0, 1.0]"


def test_categorical_descriptor_input_to_pydantic_field_falls_back_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    n = LLM_ENUM_SCHEMA_THRESHOLD + 1
    categories = [f"c{i}" for i in range(n)]
    # distinct values per category so the per-descriptor variance validator passes
    values = [[float(i)] for i in range(n)]
    feat = CategoricalDescriptorInput(
        key="big",
        categories=categories,
        descriptors=["d1"],
        values=values,
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the categories (via the prefix)
    assert "c0" in field_info.description
    assert f"c{n - 1}" in field_info.description
