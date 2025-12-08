import torch

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    ContinuousDescriptorInput,
    ContinuousInput,
    MeanFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.surrogates.engineered_features import (
    map_mean_feature,
    map_sum_feature,
    map_weighted_sum_feature,
)
from bofire.utils.torch_tools import tkwargs


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
