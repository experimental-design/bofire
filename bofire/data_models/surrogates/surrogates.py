import itertools
from typing import Annotated, List

from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs


class Surrogates(BaseModel):
    type: str
    surrogates: Annotated[List, Field(min_items=1)]

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        return {
            key: value
            for model in self.surrogates
            for key, value in model.input_preprocessing_specs.items()
        }

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.outputs.get() for model in self.surrogates]  # type: ignore
                )
            )
        )

    @validator("surrogates")
    def validate_surrogates(cls, v, values):
        # validate that all surrogates are single output surrogates
        for model in v:
            if len(model.outputs) != 1:
                raise ValueError("Only single output surrogates allowed.")
        # check that the output feature keys are distinct
        used_output_feature_keys = list(
            itertools.chain.from_iterable([model.outputs.get_keys() for model in v])
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across surrogates.")
        # get the feature keys present in all surrogates
        used_feature_keys = []
        for model in v:
            for key in model.inputs.get_keys():
                if key not in used_feature_keys:
                    used_feature_keys.append(key)
        # check that the features and preprocessing steps are equal trough the surrogates
        for key in used_feature_keys:
            features = [
                model.inputs.get_by_key(key)
                for model in v
                if key in model.inputs.get_keys()
            ]
            preproccessing = [
                model.input_preprocessing_specs[key]
                for model in v
                if key in model.input_preprocessing_specs
            ]
            if all(features) is False:
                raise ValueError(f"Features with key {key} are incompatible.")
            if all(i == preproccessing[0] for i in preproccessing) is False:
                raise ValueError(
                    f"Preprocessing steps for features with {key} are incompatible."
                )
        return v

    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        used_output_feature_keys = self.outputs.get_keys()
        if sorted(used_output_feature_keys) != sorted(outputs.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.surrogates):
            if len(model.inputs) > len(inputs):
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
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
