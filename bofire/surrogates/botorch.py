import base64
import io

import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import ChainedInputTransform, FilterFeatures

from bofire.data_models.surrogates.api import BotorchSurrogate as DataModel
from bofire.surrogates.surrogate import Surrogate
from bofire.utils.torch_tools import tkwargs


class BotorchSurrogate(Surrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            preds = self.model.posterior(X=X, observation_noise=True).mean.cpu().detach().numpy()  # type: ignore
            stds = np.sqrt(self.model.posterior(X=X, observation_noise=True).variance.cpu().detach().numpy())  # type: ignore
        return preds, stds

    @property
    def is_compatibilized(self) -> bool:
        if self.is_fitted:
            if hasattr(self.model, "input_transform"):
                if self.model.input_transform is not None:  # type: ignore
                    if isinstance(self.model.input_transform, FilterFeatures):  # type: ignore
                        return True
                    if isinstance(self.model.input_transform, ChainedInputTransform):  # type: ignore
                        if "tcompatibilize" in self.model.input_transform.keys():  # type: ignore
                            return True
        return False

    def decompatibilize(self):
        if self.is_fitted:
            if self.is_compatibilized:
                if isinstance(self.model.input_transform, FilterFeatures):  # type: ignore
                    self.model.input_transform = None  # type: ignore
                elif isinstance(self.model.input_transform, ChainedInputTransform):  # type: ignore
                    self.model.input_transform = self.model.input_transform.tf2  # type: ignore
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
        self.model = torch.load(buffer)
