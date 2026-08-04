import pytest

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.features.api import (
    ContinuousDescriptorInput,
    ContinuousInput,
    SumFeature,
    WeightedSumFeature,
)


def test_engineered_features():
    inputs = Inputs(
        features=[
            ContinuousDescriptorInput(
                key="feat1",
                bounds=(0, 1),
                descriptors=["desc1", "desc2"],
                values=[0.5, 0.5],
            ),
            ContinuousDescriptorInput(
                key="feat2",
                bounds=(0, 1),
                descriptors=["desc1", "desc3"],
                values=[0.3, 0.7],
            ),
            ContinuousInput(key="feat3", bounds=(0, 1)),
            ContinuousInput(key="feat4", bounds=(0, 1)),
        ]
    )
    engineered_features = EngineeredFeatures(
        features=[
            WeightedSumFeature(
                key="w_sum1",
                features=["feat1", "feat2"],
                descriptors=["desc1", "desc2"],
            ),
            SumFeature(key="sum1", features=["feat3", "feat4"]),
        ]
    )
    with pytest.raises(
        ValueError,
        match="Not all descriptors",
    ):
        engineered_features.validate_inputs(inputs)

    assert engineered_features.get_features2idx(offset=4) == {
        "sum1": (4,),
        "w_sum1": (5, 6),
    }

    assert engineered_features.get_feature_indices(
        offset=2,
        feature_keys=[
            "w_sum1",
        ],
    ) == [3, 4]
