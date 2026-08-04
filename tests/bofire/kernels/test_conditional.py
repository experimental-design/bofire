from typing import Callable

import pytest
import torch

from bofire.data_models.constraints.condition import (
    NonZeroCondition,
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.kernels.conditional import WedgeKernel
from bofire.data_models.kernels.continuous import LinearKernel
from bofire.kernels.conditional import (
    build_indicator_func,
    compute_base_kernel_active_dims,
)


def _get_features_to_idx_mapper(feats: list[str]) -> Callable[[list[str]], list[int]]:
    return lambda ks: list(map(feats.index, ks))


def test_build_indicator_func():
    feats = ["f1", "f2", "f3", "f4", "indicator1", "indicator2"]
    conditions = [
        ("f1", "indicator1", ThresholdCondition(threshold=1.0, operator=">")),
        ("f2", "indicator2", SelectionCondition(selection=[5, 10])),
        ("f3", "f3", NonZeroCondition()),
    ]

    indicator_func = build_indicator_func(
        conditions, _get_features_to_idx_mapper(feats)
    )

    X = torch.tensor(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # all inactive
            [0.0, 0.5, 0.0, 0.0, 1.5, 0.0],  # only f1 active
            [0.5, 0.5, 0.5, 0.0, 2.0, 5.0],  # all active
        ]
    )

    # f4, indicator1, and indicator2 are always active, since they have no conditions
    expected_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],  # all inactive
            [1, 0, 0, 1, 1, 1],  # only f1 active
            [1, 1, 1, 1, 1, 1],  # all active
        ],
        dtype=torch.bool,
    )

    assert torch.allclose(indicator_func(X), expected_mask)


@pytest.mark.parametrize(
    ["kernel_features", "expected_active_dims"],
    [
        [[], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        [["f1", "f2", "f3", "f4"], [0, 1, 2, 3, 6, 7, 8]],
        [["f2", "f4"], [1, 3, 7]],
    ],
)
def test_compute_base_kernel_active_dims(kernel_features, expected_active_dims):
    feats = ["f1", "f2", "f3", "f4", "indicator1", "indicator2"]
    conditions = [
        ("f1", "indicator1", ThresholdCondition(threshold=1.0, operator=">")),
        ("f2", "indicator2", SelectionCondition(selection=[5, 10])),
        ("f3", "f3", NonZeroCondition()),
    ]

    active_dims = list(range(len(feats)))

    # conditions are required to determine which dimensions have an embedding appended
    data_model = WedgeKernel(
        base_kernel=LinearKernel(features=kernel_features), conditions=conditions
    )

    base_kernel_active_dims = compute_base_kernel_active_dims(
        data_model,
        active_dims,
        _get_features_to_idx_mapper(feats),
    )

    assert expected_active_dims == base_kernel_active_dims


def test_compute_base_kernel_active_dims_invalid():
    feats = ["f1", "f2", "f3", "f4", "indicator1", "indicator2"]
    conditions = [
        ("f1", "indicator1", ThresholdCondition(threshold=1.0, operator=">")),
        ("f2", "indicator2", SelectionCondition(selection=[5, 10])),
        ("f3", "f3", NonZeroCondition()),
    ]

    active_dims = list(range(len(feats)))

    data_model = WedgeKernel(base_kernel=LinearKernel(), conditions=conditions)

    with pytest.raises(ValueError, match="active_dims"):
        compute_base_kernel_active_dims(
            data_model,
            [],
            _get_features_to_idx_mapper(feats),
        )

    with pytest.raises(ValueError, match="features_to_idx_mapper"):
        compute_base_kernel_active_dims(
            data_model,
            active_dims,
            None,
        )
