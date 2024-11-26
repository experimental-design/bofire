import itertools
from abc import ABC
from typing import List

import botorch
import pandas as pd
import torch
from botorch.models import ModelList
from botorch.models.transforms.input import ChainedInputTransform, FilterFeatures

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.surrogates.api import BotorchSurrogates as DataModel
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.mapper import map as map_surrogate
from bofire.surrogates.trainable import TrainableSurrogate


class BotorchSurrogates(ABC):
    surrogates: List[BotorchSurrogate]

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.surrogates = [map_surrogate(model) for model in data_model.surrogates]  # type: ignore

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return {
            key: value
            for model in self.surrogates
            for key, value in model.input_preprocessing_specs.items()
        }

    def fit(self, experiments: pd.DataFrame):
        for model in self.surrogates:
            if isinstance(model, TrainableSurrogate):
                model.fit(experiments)

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.outputs.get() for model in self.surrogates],
                ),
            ),
        )

    # TODO: is this really needed here, code duplication with functional model
    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        used_output_feature_keys = self.outputs.get_keys()
        if sorted(used_output_feature_keys) != sorted(outputs.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.surrogates):
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

    def compatibilize(self, inputs: Inputs, outputs: Outputs) -> ModelList:
        # TODO: add sync option
        # check if models are compatible to provided inputs and outputs
        # of the optimization domain
        self._check_compability(inputs=inputs, outputs=outputs)
        features2idx, _ = inputs._get_transform_info(self.input_preprocessing_specs)
        all_gp = True
        botorch_models = []
        # we sort the models by sorting them with their occurrence in outputs
        for output_feature_key in outputs.get_keys():
            # get the corresponding model
            model = {model.outputs[0].key: model for model in self.surrogates}[
                output_feature_key
            ]
            if model.model is None:
                raise ValueError(
                    f"Surrogate for output feature {output_feature_key} not fitted.",
                )
            # in case that inputs are complete we do not need to adjust anything
            if len(model.inputs) == len(inputs):
                botorch_models.append(model.model)
            # in this case we have to care for the indices
            if len(model.inputs) < len(inputs):
                indices = []
                for key in model.inputs.get_keys():
                    indices += features2idx[key]
                features_filter = FilterFeatures(
                    feature_indices=torch.tensor(indices, dtype=torch.int64),
                    transform_on_train=False,
                )
                if (
                    hasattr(model.model, "input_transform")
                    and model.model.input_transform is not None
                ):
                    model.model.input_transform = ChainedInputTransform(
                        tcompatibilize=features_filter,
                        tf2=model.model.input_transform,
                    )
                else:
                    model.model.input_transform = features_filter

                botorch_models.append(model.model)
            if isinstance(model.model, botorch.models.SingleTaskGP) is False:
                all_gp = False

        if len(botorch_models) == 1:
            return botorch_models[0]
        if all_gp:
            return botorch.models.ModelListGP(*botorch_models)
        return ModelList(*botorch_models)
