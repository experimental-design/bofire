import math

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints


def test_nchoosek_to_description():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    assert c.to_description() == "Choose 1-2 active features from ['x1', 'x2', 'x3']"


def test_nchoosek_to_description_none_valid():
    c = NChooseKConstraint(
        features=["x1", "x2"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    assert c.to_description() == "Choose 1-2 active features from ['x1', 'x2'], or none"


def test_nchoosek_count_is_valid():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    assert c.count_is_valid(1) is True
    assert c.count_is_valid(2) is True
    assert c.count_is_valid(0) is False
    assert c.count_is_valid(3) is False


def test_nchoosek_count_is_valid_none_also_valid():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    # none_also_valid carves out count==0 even though min_count > 0
    assert c.count_is_valid(0) is True
    # max_count is still a hard ceiling
    assert c.count_is_valid(3) is False


@pytest.mark.parametrize("min_count,max_count", [(0, 1), (1, 2), (1, 1), (0, 2)])
def test_nchoosek_residuals_with_experiment_metadata(min_count, max_count):
    index = pd.Index(["batch-B", "batch-A", "batch-B", "batch-C"], name="experiment")
    experiments = pd.DataFrame(
        {
            "solvent": pd.Categorical(["water", "ethanol", "water", "ethanol"]),
            "x2": [0.0, 0.0, 0.0, 1.0],
            "timestamp": pd.date_range("2025-01-01", periods=4),
            "yield": [0.1, 0.2, 0.3, 0.4],
            "x1": [0.0, 0.001, 1.0, 1.0],
        },
        index=index,
    )
    constraint = NChooseKConstraint(
        features=["x1", "x2"],
        min_count=min_count,
        max_count=max_count,
        none_also_valid=False,
    )
    zero_counts = [2.0, 1 + math.exp(-0.5), 1.0, 0.0]
    expected = pd.Series(
        [
            max(0, 2 - max_count - zeros) + max(0, zeros - (2 - min_count))
            for zeros in zero_counts
        ],
        index=index,
    )
    pd.testing.assert_series_equal(constraint(experiments), expected, check_dtype=False)


def test_nchoosek_residual_alignment_with_other_constraints():
    experiments = pd.DataFrame(
        {"x1": [1.0, 0.0], "x2": [1.0, 0.0]},
        index=pd.Index(["exp-B", "exp-A"], name="experiment"),
    )
    constraints = Constraints(
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2"],
                min_count=1,
                max_count=1,
                none_also_valid=False,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[1, 1],
                rhs=1,
            ),
        ],
    )
    expected = pd.DataFrame(
        {0: [1.0, 1.0], 1: np.array([1.0, -1.0]) / math.sqrt(2)},
        index=experiments.index,
    )
    pd.testing.assert_frame_equal(constraints(experiments), expected)
