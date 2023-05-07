from typing import Dict, Optional

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import TanimotoGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs

from botorch.models.gp_regression import SingleTaskGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from bofire.data_models.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
)

from gpytorch.module import Module
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor

import base64
import io
import warnings


class TanimotoGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Optional[Module] = ScaleKernel(base_kernel=TanimotoKernel()),
        outcome_transform: Optional[OutcomeTransform] = None,
    ):
        super().__init__(
            train_X,
            train_Y,
            likelihood=GaussianLikelihood(),
            covar_module=covar_module,
            outcome_transform=outcome_transform,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TanimotoGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[TanimotoGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        self.model = TanimotoGP(
            train_X=tX,
            train_Y=tY,
            covar_module=self.kernel.to_gpytorch(
                batch_shape=torch.Size(),
                ard_num_dims=tX.shape[1],
                active_dims=list(range(tX.shape[1])),
            ),
            outcome_transform=Standardize(m=tY.shape[-1]),
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO()
        torch.save(self.model, buffer, pickle_module=cloudpickle_module)  # type: ignore
        return base64.b64encode(buffer.getvalue()).decode()
        # return codecs.encode(pickle.dumps(self.model), "base64").decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, pickle_module=cloudpickle_module)  # type: ignore
