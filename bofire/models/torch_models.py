import itertools
from typing import List, Literal, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.models import ModelList
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model as BotorchBaseModel
from botorch.models.transforms.input import ChainedInputTransform, FilterFeatures
from pydantic import validator

from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    InputFeatures,
    NumericalInput,
    OutputFeatures,
    TInputTransformSpecs,
)
from bofire.domain.util import PydanticBaseModel
from bofire.models.model import Model, TrainableModel
from bofire.utils.enum import CategoricalEncodingEnum
from bofire.utils.torch_tools import tkwargs


class BotorchModel(Model):

    model: Optional[BotorchBaseModel]

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        input_features = values["input_features"]
        categorical_keys = input_features.get_keys(CategoricalInput, exact=True)
        descriptor_keys = input_features.get_keys(
            CategoricalDescriptorInput, exact=True
        )
        for key in categorical_keys:
            if (
                v.get(key, CategoricalEncodingEnum.ONE_HOT)
                != CategoricalEncodingEnum.ONE_HOT
            ):
                raise ValueError(
                    "Botorch based models have to use one hot encodings for categoricals"
                )
            else:
                v[key] = CategoricalEncodingEnum.ONE_HOT
        # TODO: include descriptors into probabilistic reparam via OneHotToDescriptor input transform
        for key in descriptor_keys:
            if v.get(key, CategoricalEncodingEnum.DESCRIPTOR) not in [
                CategoricalEncodingEnum.DESCRIPTOR,
                CategoricalEncodingEnum.ONE_HOT,
            ]:
                raise ValueError(
                    "Botorch based models have to use one hot encodings or descriptor encodings for categoricals."
                )
            elif v.get(key) is None:
                v[key] = CategoricalEncodingEnum.DESCRIPTOR
        for key in input_features.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError(
                    "Botorch based models have to use internal transforms to preprocess numerical features."
                )
        return v

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            preds = self.model.posterior(X=X, observation_noise=True).mean.cpu().detach().numpy()  # type: ignore
            stds = np.sqrt(self.model.posterior(X=X, observation_noise=True).variance.cpu().detach().numpy())  # type: ignore
        return preds, stds


class BotorchModels(PydanticBaseModel):

    models: List[BotorchModel]

    @validator("models")
    def validate_models(cls, v, values):
        # validate that all models are single output models
        # TODO: this restriction has to be removed at some point
        for model in v:
            if len(model.output_features) != 1:
                raise ValueError("Only single output models allowed.")
        # check that the output feature keys are distinct
        used_output_feature_keys = list(
            itertools.chain.from_iterable(
                [model.output_features.get_keys() for model in v]
            )
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across models.")
        # get the feature keys present in all models
        used_feature_keys = []
        for model in v:
            for key in model.input_features.get_keys():
                if key not in used_feature_keys:
                    used_feature_keys.append(key)
        # check that the features and preprocessing steps are equal trough the models
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

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        return {
            key: value
            for model in self.models
            for key, value in model.input_preprocessing_specs.items()
        }

    def fit(self, experiments: pd.DataFrame):
        for model in self.models:
            if isinstance(model, TrainableModel):
                model.fit(experiments)

    @property
    def output_features(self) -> OutputFeatures:
        return OutputFeatures(
            features=list(
                itertools.chain.from_iterable(
                    [model.output_features.get() for model in self.models]  # type: ignore
                )
            )
        )

    def _check_compability(
        self, input_features: InputFeatures, output_features: OutputFeatures
    ):
        # TODO: add sync option
        used_output_feature_keys = self.output_features.get_keys()
        if sorted(used_output_feature_keys) != sorted(output_features.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.models):
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
        self, input_features: InputFeatures, output_features: OutputFeatures
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
            model = {model.output_features[0].key: model for model in self.models}[
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
                        tf1=features_filter, tf2=model.model.input_transform  # type: ignore
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


class EmpiricalModel(BotorchModel):
    """All necessary functions has to be implemented in the model which can then be loaded
    from cloud pickle.

    Attributes:
        model (DeterministicModel): Botorch model instance.
    """

    type: Literal["EmpiricalModel"] = "EmpiricalModel"
    model: Optional[DeterministicModel] = None
