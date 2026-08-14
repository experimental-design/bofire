import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.features.descriptors import Descriptors


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
    t_samples = c.to_encoding(OneHotEncoding(), samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            columns=[f"{key}_{cat_str}" for cat_str in categories],
        ),
    )
    untransformed = c.from_encoding(OneHotEncoding(), t_samples)
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
    samples = c.from_encoding(OneHotEncoding(), one_hot_values)
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
        c.from_encoding(OneHotEncoding(), one_hot_values)


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
    t_samples = c.to_encoding(OneHotEncoding(drop_first=True), samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            columns=[f"{key}_{cat_str}" for cat_str in categories[1:]],
        ),
    )
    untransformed = c.from_encoding(OneHotEncoding(drop_first=True), t_samples)
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
    samples = c.from_encoding(OneHotEncoding(drop_first=True), one_hot_values)
    assert np.all(samples == pd.Series([categories[1], categories[0]]))


@pytest.mark.parametrize(
    "feature, transform_type, values, expected",
    [
        (
            CategoricalInput(key="c", categories=["B", "A", "C"]),
            OneHotEncoding(),
            None,
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            CategoricalInput(
                key="c",
                categories=["B", "A", "C"],
                allowed=[True, False, True],
            ),
            OneHotEncoding(),
            pd.Series(["A", "B", "C"]),
            ([0, 0, 0], [1, 1, 1]),
        ),
        (
            CategoricalInput(
                key="c",
                categories=["B", "A", "C"],
                allowed=[True, False, True],
            ),
            OneHotEncoding(),
            None,
            ([0, 0, 0], [1, 0, 1]),
        ),
        (
            CategoricalInput(key="c", categories=["B", "A", "C"]),
            OneHotEncoding(drop_first=True),
            None,
            ([0, 0], [1, 1]),
        ),
    ],
)
def test_one_hot_get_bounds(feature, transform_type, values, expected):
    lower, upper = feature.get_bounds(transform_type=transform_type, values=values)
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


@pytest.mark.parametrize(
    "encoding",
    [OneHotEncoding(), OneHotEncoding(drop_first=True), OrdinalEncoding()],
)
def test_descriptor_data_does_not_reach_non_descriptor_bounds(encoding):
    """An encoding that ignores descriptors must not change when a feature carries some."""
    plain = CategoricalInput(key="c", categories=["B", "A", "C"])
    described = CategoricalInput(
        key="c",
        categories=["B", "A", "C"],
        descriptors=Descriptors(columns={"alpha": [1, 3, 5], "beta": [2, 4, 6]}),
    )
    assert plain.get_bounds(transform_type=encoding) == described.get_bounds(
        transform_type=encoding
    )
