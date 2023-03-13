import itertools
from abc import ABC
from typing import List

import botorch
import pandas as pd
import torch
from botorch.models import ModelList
from botorch.models.transforms.input import ChainedInputTransform, FilterFeatures

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.surrogates.api import BotorchSurrogates as DataModel
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
        self.surrogates = [map_surrogate(model) for model in data_model.surrogates]

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
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

    def compatibilize(
        self, input_features: Inputs, output_features: Outputs
    ) -> ModelList:
        # TODO: add sync option
        # check if models are compatible to provided inputs and outputs
        # of the optimization domain
        self._check_compability(
            input_features=input_features, output_features=output_features
        )
        features2idx, _ = input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        #
        all_gp = True
        botorch_models = []
        # we sort the models by sorting them with their occurence in output_features
        for output_feature_key in output_features.get_keys():
            # get the corresponding model
            model = {model.output_features[0].key: model for model in self.surrogates}[
                output_feature_key
            ]
            # in case that inputs are complete we do not need to adjust anything
            if len(model.input_features) == len(input_features):
                botorch_models.append(model.model)
            # in this case we have to care for the indices
            if len(model.input_features) < len(input_features):
                indices = []
                for key in model.input_features.get_keys():
                    indices += features2idx[key]
                features_filter = FilterFeatures(
                    feature_indices=torch.tensor(indices, dtype=torch.int64),
                    transform_on_train=False,
                )
                if (
                    hasattr(model.model, "input_transform")
                    and model.model.input_transform is not None  # type: ignore
                ):
                    model.model.input_transform = ChainedInputTransform(  # type: ignore
                        tcompatibilize=features_filter, tf2=model.model.input_transform  # type: ignore
                    )
                else:
                    model.model.input_transform = features_filter  # type: ignore

                botorch_models.append(model.model)
                if isinstance(model.model, botorch.models.SingleTaskGP) is False:
                    all_gp = False

        if len(botorch_models) == 1:
            return botorch_models[0]
        if all_gp:
            return botorch.models.ModelListGP(*botorch_models)
        return ModelList(*botorch_models)
