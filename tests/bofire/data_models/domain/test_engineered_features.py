import pytest

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.features._descriptors import Descriptors
from bofire.data_models.features.api import (
    ContinuousInput,
    SumFeature,
    WeightedSumFeature,
)


def test_engineered_features():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="feat1",
                bounds=(0, 1),
                descriptors=Descriptors(columns={"desc1": [0.5], "desc2": [0.5]}),
            ),
            ContinuousInput(
                key="feat2",
                bounds=(0, 1),
                descriptors=Descriptors(columns={"desc1": [0.3], "desc3": [0.7]}),
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
                columns=["desc1", "desc2"],
            ),
            SumFeature(key="sum1", features=["feat3", "feat4"]),
        ]
    )
    with pytest.raises(
        ValueError,
        match="descriptor columns .* are not available",
    ):
        engineered_features.validate_inputs(inputs)

    # index bookkeeping needs components that actually agree, so ask against a valid set
    # rather than the one whose validation just failed
    valid_inputs = Inputs(
        features=[
            ContinuousInput(
                key=key,
                bounds=(0, 1),
                descriptors=Descriptors(columns={"desc1": [0.5], "desc2": [0.5]}),
            )
            for key in ("feat1", "feat2")
        ]
        + [ContinuousInput(key=key, bounds=(0, 1)) for key in ("feat3", "feat4")]
    )
    engineered_features.validate_inputs(valid_inputs)

    assert engineered_features.get_features2idx(valid_inputs, offset=4) == {
        "sum1": (4,),
        "w_sum1": (5, 6),
    }

    assert engineered_features.get_feature_indices(
        valid_inputs,
        offset=2,
        feature_keys=[
            "w_sum1",
        ],
    ) == [3, 4]
