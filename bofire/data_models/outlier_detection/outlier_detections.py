import itertools
from typing import List

from pydantic import validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.outlier_detection.outlier_detection import IterativeTrimming

OutlierDetectors = IterativeTrimming


class OutlierDetections(BaseModel):
    """ "List of Outlier detectors.

    Behaves similar to a outlier_detector."""

    detectors: List[OutlierDetectors]

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.base_gp.outputs.get() for model in self.detectors]  # type: ignore
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
            if len(model.base_gp.inputs) > len(inputs):  # type: ignore
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
                )
            for feat in model.base_gp.inputs:  # type: ignore
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

    @validator("detectors")
    def validate_base_gps(cls, v, values):
        # validate that all base_gps are single output surrogates
        # TODO: this restriction has to be removed at some point
        for model in v:
            if len(model.base_gp.outputs) != 1:
                raise ValueError("Only single output base_gps allowed.")
        # check that the output feature keys are distinctw
        used_output_feature_keys = list(
            itertools.chain.from_iterable(
                [model.base_gp.outputs.get_keys() for model in v]
            )
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across surrogates.")
        # get the feature keys present in all surrogates
        used_feature_keys = []
        for model in v:
            for key in model.base_gp.inputs.get_keys():
                if key not in used_feature_keys:
                    used_feature_keys.append(key)
        # check that the features and preprocessing steps are equal trough the surrogates
        for key in used_feature_keys:
            features = [
                model.base_gp.inputs.get_by_key(key)
                for model in v
                if key in model.base_gp.inputs.get_keys()
            ]
            preproccessing = [
                model.base_gp.input_preprocessing_specs[key]
                for model in v
                if key in model.base_gp.input_preprocessing_specs
            ]
            if all(features) is False:
                raise ValueError(f"Features with key {key} are incompatible.")
            if len(set(preproccessing)) > 1:
                raise ValueError(
                    f"Preprocessing steps for features with {key} are incompatible."
                )
        return v
