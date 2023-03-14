import itertools
from typing import List

from pydantic import validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.surrogates.botorch import BotorchSurrogate


class BotorchSurrogates(BaseModel):
    """ "List of botorch surrogates.

    Behaves similar to a Surrogate."""

    surrogates: List[BotorchSurrogate]

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        return {
            key: value
            for model in self.surrogates
            for key, value in model.input_preprocessing_specs.items()
        }

    @property
    def output_features(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.output_features.get() for model in self.surrogates]  # type: ignore
                )
            )
        )

    def _check_compability(self, input_features: Inputs, output_features: Outputs):
        # TODO: add sync option
        used_output_feature_keys = self.output_features.get_keys()
        if sorted(used_output_feature_keys) != sorted(output_features.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.surrogates):
            if len(model.input_features) > len(input_features):
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
                )
            for feat in model.input_features:
                try:
                    other_feat = input_features.get_by_key(feat.key)
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
        if len(used_feature_keys) != len(input_features):
            raise ValueError("Unused features are present.")

    @validator("surrogates")
    def validate_surrogates(cls, v, values):
        # validate that all surrogates are single output surrogates
        # TODO: this restriction has to be removed at some point
        for model in v:
            if len(model.output_features) != 1:
                raise ValueError("Only single output surrogates allowed.")
        # check that the output feature keys are distinctw
        used_output_feature_keys = list(
            itertools.chain.from_iterable(
                [model.output_features.get_keys() for model in v]
            )
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across surrogates.")
        # get the feature keys present in all surrogates
        used_feature_keys = []
        for model in v:
            for key in model.input_features.get_keys():
                if key not in used_feature_keys:
                    used_feature_keys.append(key)
        # check that the features and preprocessing steps are equal trough the surrogates
        for key in used_feature_keys:
            features = [
                model.input_features.get_by_key(key)
                for model in v
                if key in model.input_features.get_keys()
            ]
            preproccessing = [
                model.input_preprocessing_specs[key]
                for model in v
                if key in model.input_preprocessing_specs
            ]
            if all(features) is False:
                raise ValueError(f"Features with key {key} are incompatible.")
            if len(set(preproccessing)) > 1:
                raise ValueError(
                    f"Preprocessing steps for features with {key} are incompatible."
                )
        return v