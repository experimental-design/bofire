from typing import Dict, Optional

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import PiecewiseLinearGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import SortTransform, tkwargs


class SortingGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.shape_kernel = data_model.shape_kernel
        self.continuous_kernel = data_model.continuous_kernel
        self.noise_prior = data_model.noise_prior
        self.outputscale_prior = data_model.outputscale_prior
        self.ard = data_model.ard

        idx_x = [data_model.inputs.get_keys().index(k) for k in data_model.x_keys]
        idx_y = [data_model.inputs.get_keys().index(k) for k in data_model.y_keys]

        sort = SortTransform(
            idx_x=idx_x,
            idx_y=idx_y,
            keep_original=False,
        )

        self.idx_shape = list(range(len(idx_x) + len(idx_y)))
        if self.ard:
            self.ard_dims = len(data_model.x_keys) + len(data_model.y_keys)
        else:
            self.ard_dims = 1

        # get indices of x keys and normalize them
        self.idx_continuous = sorted(
            [data_model.inputs.get_keys().index(k) for k in data_model.continuous_keys]
        )

        if len(self.idx_continuous) > 0:
            bounds = torch.tensor(
                data_model.inputs.get_by_keys(data_model.continuous_keys).get_bounds(
                    specs={},
                ),
            ).to(**tkwargs)
            norm = Normalize(
                indices=self.idx_continuous,
                d=len(data_model.inputs.get_keys()),
                bounds=bounds,
            )

        self.transform = ChainedInputTransform(tf1=sort, tf2=norm)

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )
        if self.continuous_kernel is not None:
            covar_module = ScaleKernel(
                base_kernel=kernels.map(
                    self.continuous_kernel,
                    active_dims=self.idx_continuous,
                    ard_num_dims=1,
                    batch_shape=torch.Size(),
                    features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                        self.input_preprocessing_specs, feats
                    ),
                )
                * kernels.map(
                    self.shape_kernel,
                    active_dims=self.idx_shape,
                    ard_num_dims=self.ard_dims,
                    batch_shape=torch.Size(),
                    features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                        self.input_preprocessing_specs, feats
                    ),
                ),
                outputscale_prior=priors.map(self.outputscale_prior),
            )
        else:
            covar_module = ScaleKernel(
                base_kernel=kernels.map(
                    self.shape_kernel,
                    active_dims=self.idx_shape,
                    ard_num_dims=self.ard_dims,
                    batch_shape=torch.Size(),
                    features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                        self.input_preprocessing_specs, feats
                    ),
                ),
                outputscale_prior=priors.map(self.outputscale_prior),
            )

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=covar_module,
            outcome_transform=(Standardize(m=tY.shape[-1])),
            input_transform=self.transform,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
