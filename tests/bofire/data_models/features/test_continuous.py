import math
import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.features.api import ContinuousDescriptorInput, ContinuousInput


def test_continuous_input_invalid_stepsize():
    with pytest.raises(
        ValueError, match="Stepsize cannot be provided for a fixed continuous input."
    ):
        ContinuousInput(key="a", bounds=(0, 0), stepsize=0.3)
    with pytest.raises(ValueError, match="Stepsize is too big for provided range."):
        ContinuousInput(key="a", bounds=(0, 1), stepsize=1.2)


def test_continuous_input_round():
    feature = ContinuousInput(key="a", bounds=(0, 5))
    values = pd.Series([1.0, 1.3, 0.55, 4.9])
    assert_series_equal(values, feature.round(values))
    feature = ContinuousInput(key="a", bounds=(0, 5), stepsize=0.25)
    assert_series_equal(pd.Series([1.0, 1.25, 0.5, 5]), feature.round(values))
    feature = ContinuousInput(key="a", bounds=(0, 5), stepsize=0.1)
    assert_series_equal(pd.Series([1.0, 1.3, 0.5, 4.9]), feature.round(values))
    # for not matching interval
    values = pd.Series([0.4, 2.06])
    feature = ContinuousInput(key="a", bounds=(0, 2.1), stepsize=0.5)
    assert_series_equal(pd.Series([0.5, 2.1]), feature.round(values))


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            ContinuousInput(key="if1", bounds=(0.5, 4)),
            (0.5, 4.0),
        ),
        (ContinuousInput(key="if1", bounds=(2.5, 2.9)), (1, 3.0)),
        (ContinuousInput(key="if2", bounds=(1, 3)), (1, 3.0)),
        (ContinuousInput(key="if2", bounds=(1, 1)), (1, 1.0)),
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


def test_continuous_input_feature_get_bounds_local():
    feat = ContinuousInput(key="if2", bounds=(0, 1), local_relative_bounds=(0.2, 0.3))
    lower, upper = feat.get_bounds(reference_value=0.3)
    assert np.isclose(lower[0], 0.1)
    assert np.isclose(upper[0], 0.6)
    # half left
    feat = ContinuousInput(
        key="if2",
        bounds=(0, 1),
        local_relative_bounds=(math.inf, 0.3),
    )
    lower, upper = feat.get_bounds(reference_value=0.3)
    assert np.isclose(lower[0], 0.0)
    assert np.isclose(upper[0], 0.6)
    # half right
    feat = ContinuousInput(
        key="if2",
        bounds=(0, 1),
        local_relative_bounds=(0.2, math.inf),
    )
    lower, upper = feat.get_bounds(reference_value=0.3)
    assert np.isclose(lower[0], 0.1)
    assert np.isclose(upper[0], 1.0)
    # fixed feature
    feat = ContinuousInput(key="if2", bounds=(1, 1), local_relative_bounds=(0.2, 0.3))
    lower, upper = feat.get_bounds(reference_value=0.3)
    assert np.isclose(lower[0], 1)
    assert np.isclose(upper[0], 1)
    with pytest.raises(
        ValueError,
        match="Only one can be used, `local_value` or `values`.",
    ):
        feat.get_bounds(reference_value=0.3, values=pd.Series([0.1, 0.2], name="if2"))


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
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
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.0, "mama"]),
            True,
        ),
        (
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.0, "mama"]),
            False,
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
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
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
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
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.1, "a"]),
        ),
        (
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([2.9, 4.0]),
        ),
        (
            specs.features.valid(ContinuousInput).obj(),
            pd.Series([4.0, 6]),
        ),
        (
            specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([3.1, 3.2, 3.4]),
        ),
    ],
)
def test_continuous_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


def test_continuous_input_is_fulfilled():
    feature = ContinuousInput(key="a", bounds=(0, 2))
    values = pd.Series([-1.0, 1.0, 2.0, 3.0], index=["a1", "a2", "a3", "a4"])
    fulfilled = feature.is_fulfilled(values)
    assert_series_equal(
        fulfilled, pd.Series([False, True, True, False], index=["a1", "a2", "a3", "a4"])
    )


@pytest.mark.parametrize(
    "feature, xt, expected",
    [
        (
            ContinuousInput(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 1, 11)),
            np.linspace(0, 10, 11),
        ),
        (
            ContinuousInput(key="a", bounds=(-10, 20)),
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
            ContinuousInput(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            ContinuousInput(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            True,
        ),
        (
            ContinuousInput(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            False,
        ),
        (
            ContinuousInput(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            False,
        ),
        (
            ContinuousInput(key="a", bounds=(0, 9)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            ContinuousInput(key="a", bounds=(0, 9)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 10 / 9, 11),
            False,
        ),
    ],
)
def test_continuous_input_feature_to_unit_range(feature, x, expected, real):
    xt = feature.to_unit_range(x)
    assert np.allclose(xt.values, expected, real)


# TODO: tidy up the continuous descriptor input stuff
@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (ContinuousInput(key="k", bounds=(1, 1)), True, [1]),
        (ContinuousInput(key="k", bounds=(1, 2)), False, None),
        (ContinuousInput(key="k", bounds=(2, 3)), False, None),
        (
            ContinuousDescriptorInput(
                key="k",
                bounds=(1, 1),
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            True,
            [1],
        ),
        (
            ContinuousDescriptorInput(
                key="k",
                bounds=(1, 2),
                descriptors=["a", "b"],
                values=[1, 2],
            ),
            False,
            None,
        ),
        (
            ContinuousDescriptorInput(
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
def test_continuous_input_feature_is_fixed(input_feature, expected, expected_value):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value
