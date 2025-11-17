import base64
import io
from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputTransform,
)
from botorch.models.transforms.outcome import OutcomeTransform, Standardize

from bofire.data_models.features.categorical import CategoricalOutput
from bofire.data_models.surrogates.api import BotorchSurrogate as DataModel
from bofire.data_models.surrogates.api import (
    TrainableBotorchSurrogate as TrainableDataModel,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_input_transform, get_scaler
from bofire.utils.torch_tools import tkwargs


class BotorchSurrogate(Surrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.categorical_encodings: InputTransformSpecs = (
            data_model.categorical_encodings
        )
        super().__init__(data_model=data_model, **kwargs)

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            preds = (
                self.model.posterior(X=X, observation_noise=True)
                .mean.cpu()
                .detach()
                .numpy()
            )
            stds = np.sqrt(
                self.model.posterior(X=X, observation_noise=True)
                .variance.cpu()
                .detach()
                .numpy(),
            )
        return preds, stds

    @property
    def is_compatibilized(self) -> bool:
        if self.is_fitted:
            if hasattr(self.model, "input_transform"):
                if self.model.input_transform is not None:
                    if isinstance(self.model.input_transform, FilterFeatures):
                        return True
                    if isinstance(self.model.input_transform, ChainedInputTransform):
                        if "tcompatibilize" in self.model.input_transform.keys():
                            return True
        return False

    def decompatibilize(self):
        if self.is_fitted:
            if self.is_compatibilized:
                if isinstance(self.model.input_transform, FilterFeatures):
                    self.model.input_transform = None
                elif isinstance(self.model.input_transform, ChainedInputTransform):
                    self.model.input_transform = self.model.input_transform.tf2
                else:
                    raise ValueError("Undefined input transform structure detected.")

    def _prepare_for_dump(self):
        """Decompatibilize the model before the dump"""
        self.decompatibilize()

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        # empty internal caches to get smaller dumps
        self.model.prediction_strategy = None
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, weights_only=False)


class TrainableBotorchSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: TrainableDataModel,
        **kwargs,
    ):
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        scaler = get_scaler(self.inputs, self.categorical_encodings, self.scaler, X)
        input_transform = get_input_transform(
            self.inputs, scaler, self.categorical_encodings
        )
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        # in case of classification we need to convert y from str to int
        if isinstance(self.outputs[0], CategoricalOutput):
            label_mapping = self.outputs[0].objective.to_dict_label()
            Y = pd.DataFrame.from_dict(
                {col: Y[col].map(label_mapping) for col in Y.columns},
            )
        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )
        # todo, we should implement log transforms also here
        outcome_transform = (
            Standardize(m=tY.shape[-1])
            if self.output_scaler == ScalerEnum.STANDARDIZE
            else None
        )
        self._fit_botorch(tX, tY, input_transform, outcome_transform, **kwargs)

    @abstractmethod
    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        pass
