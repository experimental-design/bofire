import pytest

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    ContinuousDescriptorInput,
    ContinuousInput,
    SumFeature,
    WeightedSumFeature,
)


def test_engineered_feature_validation():
    sum_feature = SumFeature(key="sum1", features=["feat1", "feat2", "feat3"])
    inputs = Inputs(
        features=[
            ContinuousInput(key="feat1", bounds=(0, 1)),
            ContinuousInput(key="feat2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(
        ValueError, match="The following features are missing in inputs:"
    ):
        sum_feature.validate_features(inputs)


def test_weighted_sum_feature_validation():
    weighted_sum_feature = WeightedSumFeature(
        key="w_sum1", features=["feat1", "feat2"], descriptors=["desc1", "desc2"]
    )
    inputs = Inputs(
        features=[
            ContinuousDescriptorInput(
                key="feat1",
                bounds=(0, 1),
                descriptors=["desc1", "desc2"],
                values=[0.5, 0.5],
            ),
            ContinuousInput(key="feat2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(
        ValueError, match="Feature 'feat2' is not a ContinuousDescriptorInput"
    ):
        weighted_sum_feature.validate_features(inputs)

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
                values=[0.5, 0.5],
            ),
        ]
    )
    with pytest.raises(ValueError, match="Not all descriptors"):
        weighted_sum_feature.validate_features(inputs)
