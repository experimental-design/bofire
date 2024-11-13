import itertools
from typing import Annotated, List

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.outlier_detection.outlier_detection import IterativeTrimming


AnyOutlierDetector = IterativeTrimming


class OutlierDetections(BaseModel):
    """ "List of Outlier detectors.

    Behaves similar to a outlier_detector.
    """

    detectors: Annotated[List[AnyOutlierDetector], Field(min_length=1)]

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.outputs.get() for model in self.detectors],
                ),
            ),
        )

    @field_validator("detectors")
    @classmethod
    def validate_detectors(cls, v):
        used_output_feature_keys = list(
            itertools.chain.from_iterable(
                [detector.outputs.get_keys() for detector in v],
            ),
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across detectors.")
        return v

    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        # TODO: add sync option
        used_output_feature_keys = self.outputs.get_keys()
        if sorted(used_output_feature_keys) != sorted(outputs.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.detectors):
            if len(model.inputs) > len(inputs):
                raise ValueError(
                    f"Model with index {i} has more features than acceptable.",
                )
            for feat in model.inputs:
                try:
                    other_feat = inputs.get_by_key(feat.key)
                except KeyError:
                    raise ValueError(f"Feature {feat.key} not found.")
                # now compare the features
                # TODO: make more sohisticated comparisons based on the type
                # has to to be implemented in features, for the start
                # we go with __eq__
                if feat != other_feat:
                    raise ValueError(f"Features with key {feat.key} are incompatible.")
                if feat.key not in used_feature_keys:
                    used_feature_keys.append(feat.key)
        if len(used_feature_keys) != len(inputs):
            raise ValueError("Unused features are present.")
