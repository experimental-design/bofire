from typing import Dict, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import PiecewiseLinearGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import InterpolateTransform, tkwargs


class PiecewiseLinearGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.noise_prior = data_model.noise_prior
        lower, upper = data_model.interpolation_range
        new_ts = torch.from_numpy(
            np.linspace(lower, upper, data_model.n_interpolation_points)
        ).to(dtype=torch.float64)
        idx_x = [data_model.inputs.get_keys().index(k) for k in data_model.x_keys]
        idx_y = [data_model.inputs.get_keys().index(k) for k in data_model.y_keys]

        self.transform = InterpolateTransform(
            new_x=new_ts,
            idx_x=idx_x,
            idx_y=idx_y,
            prepend_x=torch.tensor(data_model.prepend_x).to(**tkwargs),
            prepend_y=torch.tensor(data_model.prepend_y).to(**tkwargs),
            append_x=torch.tensor(data_model.append_x).to(**tkwargs),
            append_y=torch.tensor(data_model.append_y).to(**tkwargs),
        )

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(tX.shape[1])),
                ard_num_dims=1,  # this keyword is ingored
            ),
            outcome_transform=(Standardize(m=tY.shape[-1])),
            input_transform=self.transform,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
