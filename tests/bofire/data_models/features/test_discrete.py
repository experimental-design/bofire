import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.features.api import DiscreteInput


@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (specs.features.valid(DiscreteInput).obj(values=[1, 2, 3]), False, None),
    ],
)
def test_discrete_input_feature_is_fixed(input_feature, expected, expected_value):
    print(input_feature)
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value


@pytest.mark.parametrize(
    "input_feature, expected_lower, expected_upper",
    [
        (
            specs.features.valid(DiscreteInput).obj(values=[1.0, 2.0, 3.0]),
            1,
            3,
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
        {"if1": [1.0, 2.0, 3.0, 4.0], "if2": [1.0, 1.0, 1.0, 1.0]},
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
            specs.features.valid(DiscreteInput).obj(values=[1, 2, 3]),
            pd.Series([random.choice([1, 2, 3]) for _ in range(20)]),
        ),
    ],
)
def test_discrete_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            specs.features.valid(DiscreteInput).obj(values=[1, 2]),
            pd.Series([1, 2, 3]),
        ),
    ],
)
def test_discrete_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


def test_discrete_input_is_fulfilled():
    feature = DiscreteInput(key="a", values=[0, 1, 2])
    values = pd.Series([-1, 0, 1, 2, 3], index=[0, 1, 2, 3, 10])
    fulfilled = feature.is_fulfilled(values)
    assert_series_equal(
        fulfilled, pd.Series([False, True, True, True, False], index=values.index)
    )


def test_from_continuous():
    d = DiscreteInput(key="d", values=[1, 2, 3])

    continuous_values = pd.DataFrame(
        columns=["d"],
        data=[1.8, 1.7, 2.9, 1.9],
    )
    samples = d.from_continuous(continuous_values)
    assert np.all(samples == pd.Series([2, 2, 3, 2]))
