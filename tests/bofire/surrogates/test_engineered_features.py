import importlib

import pandas as pd
import pytest
import torch

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    CloneFeature,
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousMolecularInput,
    MeanFeature,
    MolecularWeightedMeanFeature,
    MolecularWeightedSumFeature,
    ProductFeature,
    SumFeature,
    WeightedMeanFeature,
    WeightedSumFeature,
)
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.surrogates.engineered_features import (
    map_clone_feature,
    map_mean_feature,
    map_molecular_weighted_mean_feature,
    map_molecular_weighted_sum_feature,
    map_product_feature,
    map_sum_feature,
    map_weighted_mean_feature,
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

    # also test with a 3D tensor (batch x repeat x features)
    # also test with a 3D tensor (batch x repeat x features)
    orig_3d = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.11, 0.12, 0.13], [0.14, 0.15, 0.16]],
        ]
    ).to(**tkwargs)
    result_3d = aggregator(orig_3d)
    assert result_3d.shape == (2, 2, 4)
    assert torch.allclose(result_3d[:, :, :-1], orig_3d)
    assert torch.allclose(
        result_3d[:, :, -1],
        orig_3d[:, :, expected_idx[0]] * orig_3d[:, :, expected_idx[1]],
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
    descriptors = torch.tensor([[1, 2], [4, 5], [7, 8]]).to(**tkwargs)
    expected = torch.matmul(orig, descriptors)

    assert result.shape[0] == 2
    assert result.shape[1] == 5

    assert torch.allclose(
        result[:, :-2], torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]]).to(**tkwargs)
    )
    assert torch.allclose(result[:, -2:], expected)


def test_map_weighted_mean_feature():
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
                descriptors=["d1", "d2", "d3"],
                values=[7, 8, 9],
            ),
        ]
    )
    aggregation = WeightedMeanFeature(
        key="agg1",
        features=["x1", "x2", "x3"],
        descriptors=["d1", "d2"],
    )

    assert aggregation.n_transformed_inputs == 2

    aggregator = map_weighted_mean_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]]).to(**tkwargs)
    result = aggregator(orig)

    descriptors = torch.tensor([[1, 2], [4, 5], [7, 8]]).to(**tkwargs)
    expected = torch.matmul(orig, descriptors) / orig.sum(dim=1, keepdim=True)

    assert result.shape[0] == 2
    assert result.shape[1] == 5
    assert torch.allclose(result[:, :-2], orig)
    assert torch.allclose(result[:, -2:], expected)


def test_map_weighted_mean_feature_zero_weight_sum():
    inputs = Inputs(
        features=[
            ContinuousDescriptorInput(
                key="x1",
                bounds=[0, 1],
                descriptors=["d1", "d2"],
                values=[1, 2],
            ),
            ContinuousDescriptorInput(
                key="x2",
                bounds=[0, 1],
                descriptors=["d1", "d2"],
                values=[3, 4],
            ),
        ]
    )
    aggregation = WeightedMeanFeature(
        key="agg1",
        features=["x1", "x2"],
        descriptors=["d1", "d2"],
    )
    aggregator = map_weighted_mean_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    # Zero weights would cause division-by-zero without epsilon clamping.
    orig = torch.tensor([[0.0, 0.0], [0.1, 0.0]]).to(**tkwargs)
    result = aggregator(orig)

    assert result.shape == (2, 4)
    assert torch.allclose(result[:, :-2], orig)
    assert torch.isfinite(result[:, -2:]).all()
    assert torch.allclose(
        result[:, -2:],
        torch.tensor([[0.0, 0.0], [1.0, 2.0]]).to(**tkwargs),
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

    # test this also for a 3d batch tensor
    orig_3d = torch.tensor(
        [
            [[0.1, 0.2], [0.4, 0.1]],
            [[0.11, 0.12], [0.14, 0.15]],
        ]
    ).to(**tkwargs)
    result_3d = aggregator(orig_3d)
    assert result_3d.shape == (2, 2, 3)
    assert torch.allclose(result_3d[:, :, :-1], orig_3d)
    expected_weighted_3d = torch.matmul(orig_3d, descriptors)
    assert torch.allclose(result_3d[:, :, -1:], expected_weighted_3d)


@pytest.mark.skipif(
    not (RDKIT_AVAILABLE and MORDRED_AVAILABLE),
    reason="requires rdkit and mordred",
)
def test_map_molecular_weighted_mean_feature():
    inputs = Inputs(
        features=[
            ContinuousMolecularInput(key="m1", bounds=[0, 1], molecule="C"),
            ContinuousMolecularInput(key="m2", bounds=[0, 1], molecule="CC"),
        ]
    )
    molfeatures = MordredDescriptors(
        descriptors=["NssCH2", "ATSC2d"], ignore_3D=True, correlation_cutoff=1.0
    )
    aggregation = MolecularWeightedMeanFeature(
        key="agg1",
        features=["m1", "m2"],
        molfeatures=molfeatures,
    )

    aggregator = map_molecular_weighted_mean_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2], [0.4, 0.1]]).to(**tkwargs)
    result = aggregator(orig)

    descriptors_df = molfeatures.get_descriptor_values(pd.Series(["C", "CC"]))
    descriptors = torch.tensor(descriptors_df.values).to(**tkwargs)
    expected_weighted = torch.matmul(orig, descriptors) / orig.sum(dim=1, keepdim=True)

    assert torch.allclose(result[:, :-1], orig)
    assert torch.allclose(result[:, -1:], expected_weighted)


def test_map_clone_feature():
    inputs = Inputs(
        features=[ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(1, 6)]
    )
    # clone two non-consecutive inputs x2 and x5
    aggregation = CloneFeature(key="agg1", features=["x2", "x5"])

    aggregator = map_clone_feature(
        inputs=inputs, transform_specs={}, feature=aggregation
    )

    orig = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]).to(
        **tkwargs
    )
    result = aggregator(orig)
    assert result.shape[0] == 2
    assert result.shape[1] == 7

    assert torch.allclose(result[:, :-2], orig)
    # clones appended in the order specified: x2 then x5
    assert torch.allclose(result[:, -2], orig[:, 1])
    assert torch.allclose(result[:, -1], orig[:, 4])

    # also test with a 3D tensor (batch x repeat x features)
    orig_3d = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            [[0.11, 0.12, 0.13, 0.14, 0.15], [0.16, 0.17, 0.18, 0.19, 0.20]],
        ]
    ).to(**tkwargs)
    result_3d = aggregator(orig_3d)
    assert result_3d.shape == (2, 2, 7)
    assert torch.allclose(result_3d[:, :, :-2], orig_3d)
    assert torch.allclose(result_3d[:, :, -2], orig_3d[:, :, 1])
    assert torch.allclose(result_3d[:, :, -1], orig_3d[:, :, 4])
