import itertools
from typing import List, Union

from pydantic import validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
from bofire.data_models.surrogates.fully_bayesian import SaasSingleTaskGPSurrogate
from bofire.data_models.surrogates.linear import LinearSurrogate
from bofire.data_models.surrogates.mixed_single_task_gp import (
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mlp import MLPEnsemble
from bofire.data_models.surrogates.polynomial import PolynomialSurrogate
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

AnyBotorchSurrogate = Union[
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MLPEnsemble,
    SaasSingleTaskGPSurrogate,
    TanimotoGPSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
]


class BotorchSurrogates(BaseModel):
    """ "List of botorch surrogates.

    Behaves similar to a Surrogate."""

    surrogates: List[AnyBotorchSurrogate]

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

    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        # TODO: add sync option
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

    @validator("surrogates")
    def validate_surrogates(cls, v, values):
        # validate that all surrogates are single output surrogates
        # TODO: this restriction has to be removed at some point
        for model in v:
            if len(model.outputs) != 1:
                raise ValueError("Only single output surrogates allowed.")
        # check that the output feature keys are distinctw
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
