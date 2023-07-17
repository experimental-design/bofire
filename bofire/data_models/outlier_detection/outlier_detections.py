import itertools
from typing import List

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.outlier_detection.outlier_detection import IterativeTrimming

AnyOutlierDetector = IterativeTrimming


class OutlierDetections(BaseModel):
    """ "List of Outlier detectors.

    Behaves similar to a outlier_detector."""

    detectors: List[AnyOutlierDetector]

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.outputs.get() for model in self.detectors]  # type: ignore
                )
            )
        )

    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        # TODO: add sync option
        used_output_feature_keys = self.outputs.get_keys()
        if sorted(used_output_feature_keys) != sorted(outputs.get_keys()):  # type: ignore
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.detectors):
            if len(model.inputs) > len(inputs):  # type: ignore
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
                )
            for feat in model.inputs:  # type: ignore
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
