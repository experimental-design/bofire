import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousMolecularInput,
    InterpolateFeature,
    MeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.surrogates.engineered_features import (
    map_interpolate_feature,
    map_mean_feature,
    map_molecular_weighted_sum_feature,
    map_product_feature,
    map_sum_feature,
    map_weighted_sum_feature,
)
from bofire.utils.torch_tools import tkwargs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None
MORDRED_AVAILABLE = importlib.util.find_spec("mordred") is not None


def test_get_sum_feature():
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
        ]
    )
    aggregation = SumFeature(key="agg1", features=["x1", "x3"])

    aggregator = map_sum_feature(inputs=inputs, transform_specs={}, feature=aggregation)

    orig = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(**tkwargs)
    result = aggregator(orig)
    assert result.shape[0] == 2
    assert result.shape[1] == 4

    assert torch.allclose(result[:, :-1], orig)
    assert torch.allclose(result[:, -1], orig[:, 0] + orig[:, 2])


def test_map_mean_feature():
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
            ContinuousInput(key="x4", bounds=[0, 1]),
        ]
    )
    aggregation = MeanFeature(key="agg1", features=["x2", "x3", "x4"])

    aggregator = map_mean_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).to(**tkwargs)
    result = aggregator(orig)
    assert result.shape[0] == 2
    assert result.shape[1] == 5

    assert torch.allclose(result[:, :-1], orig)
    assert torch.allclose(result[:, -1], orig[:, 1:4].mean(dim=1))


@pytest.mark.parametrize(
    "features, expected_idx",
    [
        (["x1", "x3"], [0, 2]),
        (["x2", "x2"], [1, 1]),
    ],
)
def test_map_product_feature(features, expected_idx):
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
        ]
    )
    aggregation = ProductFeature(key="agg1", features=features)

    aggregator = map_product_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(**tkwargs)
    result = aggregator(orig)
    assert result.shape[0] == 2
    assert result.shape[1] == 4

    assert torch.allclose(result[:, :-1], orig)
    assert torch.allclose(
        result[:, -1], orig[:, expected_idx[0]] * orig[:, expected_idx[1]]
    )


def test_map_weighted_sum_feature():
    inputs = Inputs(
        features=[
            ContinuousDescriptorInput(
                key="x1",
                bounds=[0, 1],
                descriptors=["d1", "d2", "d3"],
                values=[1, 2, 3],
            ),
            ContinuousDescriptorInput(
                key="x2",
                bounds=[0, 1],
                descriptors=["d1", "d2", "d3"],
                values=[4, 5, 6],
            ),
            ContinuousDescriptorInput(
                key="x3",
                bounds=[0, 1],
                descriptors=[
                    "d1",
                    "d2",
                    "d3",
                ],
                values=[
                    7,
                    8,
                    9,
                ],
            ),
        ]
    )
    aggregation = WeightedSumFeature(
        key="agg1",
        features=["x1", "x2", "x3"],
        descriptors=[
            "d1",
            "d2",
        ],
    )

    assert aggregation.n_transformed_inputs == 2

    aggregator = map_weighted_sum_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]]).to(**tkwargs)
    result = aggregator(orig)

    assert result.shape[0] == 2
    assert result.shape[1] == 5

    assert torch.allclose(
        result[:, :-2], torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]]).to(**tkwargs)
    )


@pytest.mark.skipif(
    not (RDKIT_AVAILABLE and MORDRED_AVAILABLE),
    reason="requires rdkit and mordred",
)
def test_map_molecular_weighted_sum_feature():
    inputs = Inputs(
        features=[
            ContinuousMolecularInput(key="m1", bounds=[0, 1], molecule="C"),
            ContinuousMolecularInput(key="m2", bounds=[0, 1], molecule="CC"),
        ]
    )
    molfeatures = MordredDescriptors(
        descriptors=["NssCH2", "ATSC2d"], ignore_3D=True, correlation_cutoff=1.0
    )
    aggregation = MolecularWeightedSumFeature(
        key="agg1",
        features=["m1", "m2"],
        molfeatures=molfeatures,
    )

    assert aggregation.n_transformed_inputs == 2

    aggregator = map_molecular_weighted_sum_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    # one is filtered out due to zero variance
    assert aggregation.n_transformed_inputs == 1

    orig = torch.tensor([[0.1, 0.2], [0.4, 0.1]]).to(**tkwargs)
    result = aggregator(orig)

    descriptors_df = molfeatures.get_descriptor_values(pd.Series(["C", "CC"]))
    descriptors = torch.tensor(descriptors_df.values).to(**tkwargs)
    expected_weighted = torch.matmul(orig, descriptors)

    assert result.shape[0] == 2
    assert result.shape[1] == 3

    assert torch.allclose(result[:, :-1], orig)
    assert torch.allclose(result[:, -1:], expected_weighted)


def test_map_interpolate_feature_with_prepend_append():
    """Test interpolation with prepend and append boundary values."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=[0, 60]),
            ContinuousInput(key="x2", bounds=[0, 60]),
            ContinuousInput(key="x3", bounds=[0, 60]),
            ContinuousInput(key="y1", bounds=[0, 1]),
            ContinuousInput(key="y2", bounds=[0, 1]),
            ContinuousInput(key="y3", bounds=[0, 1]),
        ]
    )

    n_interp = 200
    feature = InterpolateFeature(
        key="interp1",
        features=["x1", "x2", "x3", "y1", "y2", "y3"],
        x_keys=["x1", "x2", "x3"],
        y_keys=["y1", "y2", "y3"],
        n_interpolation_points=n_interp,
        prepend_x=[0.0],
        append_x=[60.0],
        prepend_y=[0.0],
        append_y=[1.0],
    )

    aggregator = map_interpolate_feature(
        inputs=inputs, transform_specs={}, feature=feature
    )

    # Same data as original test: x=[10,40,55] y=[0.2,0.5,0.75] and
    # x=[10,20,55] y=[0.2,0.5,0.7]. After prepend/append:
    # x=[0,10,40,55,60], y=[0,0.2,0.5,0.75,1.0] etc.
    tX = torch.tensor([[10, 40, 55, 0.2, 0.5, 0.75], [10, 20, 55, 0.2, 0.5, 0.7]]).to(
        **tkwargs
    )
    result = aggregator(tX)

    assert result.shape == (2, 6 + n_interp)
    assert torch.allclose(result[:, :6], tX)

    x_new = np.linspace(0, 60, n_interp)
    x = np.array([[0.0, 10, 40, 55, 60], [0.0, 10, 20, 55, 60]])
    y = np.array([[0.0, 0.2, 0.5, 0.75, 1.0], [0.0, 0.2, 0.5, 0.7, 1.0]])
    y_new = np.array([np.interp(x_new, x[i], y[i]) for i in range(2)])
    np.testing.assert_allclose(result[:, 6:].numpy(), y_new, rtol=1e-6)


def test_map_interpolate_feature_no_prepend_append():
    """Test interpolation without prepend/append, all coordinates from features."""
    inputs = Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=[0, 60]) for i in range(5)]
        + [ContinuousInput(key=f"y{i}", bounds=[0, 1]) for i in range(5)]
    )

    n_interp = 200
    feature = InterpolateFeature(
        key="interp1",
        features=[f"x{i}" for i in range(5)] + [f"y{i}" for i in range(5)],
        x_keys=[f"x{i}" for i in range(5)],
        y_keys=[f"y{i}" for i in range(5)],
        n_interpolation_points=n_interp,
    )

    aggregator = map_interpolate_feature(
        inputs=inputs, transform_specs={}, feature=feature
    )

    tX = torch.tensor(
        [
            [0, 10, 40, 55, 60, 0, 0.2, 0.5, 0.75, 1],
            [0, 10, 20, 55, 60, 0, 0.2, 0.5, 0.7, 1],
        ],
    ).to(**tkwargs)
    result = aggregator(tX)

    assert result.shape == (2, 10 + n_interp)
    assert torch.allclose(result[:, :10], tX)

    x_new = np.linspace(0, 60, n_interp)
    x = np.array([[0.0, 10, 40, 55, 60], [0.0, 10, 20, 55, 60]])
    y = np.array([[0.0, 0.2, 0.5, 0.75, 1.0], [0.0, 0.2, 0.5, 0.7, 1.0]])
    y_new = np.array([np.interp(x_new, x[i], y[i]) for i in range(2)])
    np.testing.assert_allclose(result[:, 10:].numpy(), y_new, rtol=1e-6)


def test_map_interpolate_feature_asymmetric_prepend_append():
    """Test interpolation with asymmetric prepend/append (only prepend_x and append_y)."""
    inputs = Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=[0, 60]) for i in range(4)]
        + [ContinuousInput(key=f"y{i}", bounds=[0, 1]) for i in range(4)]
    )

    n_interp = 200
    feature = InterpolateFeature(
        key="interp1",
        features=[f"x{i}" for i in range(4)] + [f"y{i}" for i in range(4)],
        x_keys=[f"x{i}" for i in range(4)],
        y_keys=[f"y{i}" for i in range(4)],
        n_interpolation_points=n_interp,
        prepend_x=[0.0],
        append_y=[1.0],
    )

    aggregator = map_interpolate_feature(
        inputs=inputs, transform_specs={}, feature=feature
    )

    # After prepend/append: x=[0,10,40,55,60], y=[0,0.2,0.5,0.75,1.0]
    tX = torch.tensor(
        [
            [10, 40, 55, 60, 0, 0.2, 0.5, 0.75],
            [10, 20, 55, 60, 0, 0.2, 0.5, 0.7],
        ],
    ).to(**tkwargs)
    result = aggregator(tX)

    assert result.shape == (2, 8 + n_interp)
    assert torch.allclose(result[:, :8], tX)

    x_new = np.linspace(0, 60, n_interp)
    x = np.array([[0.0, 10, 40, 55, 60], [0.0, 10, 20, 55, 60]])
    y = np.array([[0.0, 0.2, 0.5, 0.75, 1.0], [0.0, 0.2, 0.5, 0.7, 1.0]])
    y_new = np.array([np.interp(x_new, x[i], y[i]) for i in range(2)])
    np.testing.assert_allclose(result[:, 8:].numpy(), y_new, rtol=1e-6)


def test_map_interpolate_feature_normalize_x_and_y():
    """Test interpolation with normalize_x and normalize_y enabled."""
    inputs = Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=[0, 100]) for i in range(4)]
        + [ContinuousInput(key=f"y{i}", bounds=[0, 100]) for i in range(4)]
    )

    n_interp = 6
    feature = InterpolateFeature(
        key="interp1",
        features=[f"x{i}" for i in range(4)] + [f"y{i}" for i in range(4)],
        x_keys=[f"x{i}" for i in range(4)],
        y_keys=[f"y{i}" for i in range(4)],
        n_interpolation_points=n_interp,
        prepend_x=[0.0],
        append_x=[100.0],
        prepend_y=[0.0],
        append_y=[100.0],
        normalize_y=100.0,
        normalize_x=True,
    )

    aggregator = map_interpolate_feature(
        inputs=inputs, transform_specs={}, feature=feature
    )

    tX = torch.tensor(
        [
            [20, 40, 60, 80, 20, 40, 60, 80],
            [20, 40, 60, 80, 20, 40, 60, 80],
        ]
    ).to(**tkwargs)
    result = aggregator(tX)

    assert result.shape == (2, 8 + n_interp)
    assert torch.allclose(result[:, :8], tX)

    expected = torch.linspace(0, 1, n_interp, dtype=torch.double)
    for i in range(2):
        assert torch.allclose(result[i, 8:], expected, atol=1e-6)


def test_map_interpolate_feature_3d_input():
    """Test interpolation with 3D tensor input (batch x q x features)."""
    inputs = Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=[0, 60]) for i in range(5)]
        + [ContinuousInput(key=f"y{i}", bounds=[0, 1]) for i in range(5)]
    )

    n_interp = 200
    feature = InterpolateFeature(
        key="interp1",
        features=[f"x{i}" for i in range(5)] + [f"y{i}" for i in range(5)],
        x_keys=[f"x{i}" for i in range(5)],
        y_keys=[f"y{i}" for i in range(5)],
        n_interpolation_points=n_interp,
    )

    aggregator = map_interpolate_feature(
        inputs=inputs, transform_specs={}, feature=feature
    )

    # 3D input: batch=1, q=2, features=10
    tX = torch.tensor(
        [
            [
                [0, 10, 40, 55, 60, 0, 0.2, 0.5, 0.75, 1],
                [0, 10, 20, 55, 60, 0, 0.2, 0.5, 0.7, 1],
            ]
        ],
    ).to(**tkwargs)
    result = aggregator(tX)

    assert result.shape == (1, 2, 10 + n_interp)
    assert torch.allclose(result[:, :, :10], tX)

    x_new = np.linspace(0, 60, n_interp)
    x = np.array([[0.0, 10, 40, 55, 60], [0.0, 10, 20, 55, 60]])
    y = np.array([[0.0, 0.2, 0.5, 0.75, 1.0], [0.0, 0.2, 0.5, 0.7, 1.0]])
    y_new = np.array([np.interp(x_new, x[i], y[i]) for i in range(2)])
    np.testing.assert_allclose(result[0, :, 10:].numpy(), y_new, rtol=1e-6)
