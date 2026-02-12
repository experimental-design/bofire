from typing import Dict, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import ChainedInputTransform, Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.kernels.api import (
    ExactWassersteinKernel as ExactWassersteinKernelData,
)

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
        self.shape_kernel = data_model.shape_kernel
        self.continuous_kernel = data_model.continuous_kernel

        self.noise_prior = data_model.noise_prior
        self.outputscale_prior = data_model.outputscale_prior
        self.ard = data_model.ard
        self.saas = data_model.saas

        lower, upper = data_model.interpolation_range
        new_ts = torch.from_numpy(
            np.linspace(lower, upper, data_model.n_interpolation_points),
        ).to(dtype=torch.float64)
        idx_x = [data_model.inputs.get_keys().index(k) for k in data_model.x_keys]
        idx_y = [data_model.inputs.get_keys().index(k) for k in data_model.y_keys]
        self.idx_x = idx_x
        self.idx_y = idx_y

        self.prepend_x = torch.tensor(data_model.prepend_x).to(**tkwargs)
        self.prepend_y = torch.tensor(data_model.prepend_y).to(**tkwargs)
        self.append_x = torch.tensor(data_model.append_x).to(**tkwargs)
        self.append_y = torch.tensor(data_model.append_y).to(**tkwargs)
        self.normalize_y = torch.tensor(data_model.normalize_y).to(**tkwargs)

        inter = InterpolateTransform(
            new_x=new_ts,
            idx_x=idx_x,
            idx_y=idx_y,
            prepend_x=torch.tensor(data_model.prepend_x).to(**tkwargs),
            prepend_y=torch.tensor(data_model.prepend_y).to(**tkwargs),
            append_x=torch.tensor(data_model.append_x).to(**tkwargs),
            append_y=torch.tensor(data_model.append_y).to(**tkwargs),
            normalize_y=torch.tensor(data_model.normalize_y).to(**tkwargs),
            normalize_x=True,
            keep_original=True,
        )

        self.idx_shape = list(range(new_ts.shape[0]))

        self.idx_continuous = sorted(
            [
                data_model.inputs.get_keys().index(k) + new_ts.shape[0]
                for k in data_model.continuous_keys
            ],
        )

        if len(self.idx_continuous) > 0:
            bounds = torch.tensor(
                data_model.inputs.get_by_keys(data_model.continuous_keys).get_bounds(
                    specs={},
                ),
            ).to(**tkwargs)
            norm = Normalize(
                indices=self.idx_continuous,
                d=len(data_model.inputs.get_keys()) + new_ts.shape[0],
                bounds=bounds,
            )

            self.transform = ChainedInputTransform(tf1=inter, tf2=norm)
        else:
            self.transform = inter

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
            # covar_module = ScaleKernel(
            #     base_kernel=kernels.map(
            #         self.continuous_kernel,
            #         active_dims=self.idx_continuous,
            #         ard_num_dims=1,
            #         batch_shape=torch.Size(),
            #         features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
            #             self.input_preprocessing_specs, feats
            #         ),
            #     )
            #     * kernels.map(
            #         self.shape_kernel,
            #         active_dims=self.idx_shape,
            #         ard_num_dims=len(self.idx_shape) if self.ard else 1,
            #         batch_shape=torch.Size(),
            #         features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
            #             self.input_preprocessing_specs, feats
            #         ),
            #     ),
            #     outputscale_prior=priors.map(self.outputscale_prior),
            # )
            covar_module = kernels.map(
                self.continuous_kernel,
                active_dims=self.idx_continuous,
                ard_num_dims=1,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            ) * kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                ard_num_dims=len(self.idx_shape) if self.ard else None,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )
        else:
            # covar_module = ScaleKernel(
            #     base_kernel=kernels.map(
            #         self.shape_kernel,
            #         active_dims=self.idx_shape,
            #         ard_num_dims=len(self.idx_shape) if self.ard else 1,
            #         batch_shape=torch.Size(),
            #         features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
            #             self.input_preprocessing_specs, feats
            #         ),
            #     ),
            #     outputscale_prior=priors.map(self.outputscale_prior),
            # )
            covar_module = kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                ard_num_dims=len(self.idx_shape) if self.ard else None,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )

        if self.saas and self.continuous_kernel is None:
            self.model = botorch.models.map_saas.AdditiveMapSaasSingleTaskGP(
                train_X=tX,
                train_Y=tY,
                outcome_transform=(Standardize(m=tY.shape[-1])),
                input_transform=self.transform,
            )  # type: ignore
        elif self.saas and self.continuous_kernel is not None:
            raise ValueError(
                "SAAS is only implemented for shape-only models without continuous kernel."
            )
        else:
            self.model = botorch.models.SingleTaskGP(  # type: ignore
                train_X=tX,
                train_Y=tY,
                covar_module=covar_module,
                outcome_transform=(Standardize(m=tY.shape[-1])),
                input_transform=self.transform,
            )

        # self.model = botorch.models.fully_bayesian.FullyBayesianSingleTaskGP(
        #     train_X=tX,
        #     train_Y=tY,
        #     outcome_transform=(Standardize(m=tY.shape[-1])),
        #     input_transform=self.transform,
        #     )  # type: ignore

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore
        self.model.likelihood.noise_covar.raw_noise_constraint = GreaterThan(5e-4)  # type: ignore

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
        # botorch.fit_fully_bayesian_model_nuts(
        #     self.model,  # type: ignore
        # )


class ExactPiecewiseLinearGPSurrogate(BotorchSurrogate, TrainableSurrogate):
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
        self.saas = data_model.saas

        idx_x = [data_model.inputs.get_keys().index(k) for k in data_model.x_keys]
        idx_y = [data_model.inputs.get_keys().index(k) for k in data_model.y_keys]

        self.idx_x = idx_x
        self.idx_y = idx_y

        self.prepend_x = torch.tensor(data_model.prepend_x).to(**tkwargs)
        self.prepend_y = torch.tensor(data_model.prepend_y).to(**tkwargs)
        self.append_x = torch.tensor(data_model.append_x).to(**tkwargs)
        self.append_y = torch.tensor(data_model.append_y).to(**tkwargs)
        self.normalize_y = torch.tensor(data_model.normalize_y).to(**tkwargs)

        if isinstance(self.shape_kernel, ExactWassersteinKernelData):
            self.shape_kernel.idx_x = list(range(len(idx_x)))
            self.shape_kernel.idx_y = list(range(len(idx_x), len(idx_x) + len(idx_y)))
            self.shape_kernel.prepend_x = data_model.prepend_x
            self.shape_kernel.prepend_y = data_model.prepend_y
            self.shape_kernel.append_x = data_model.append_x
            self.shape_kernel.append_y = data_model.append_y
            self.shape_kernel.normalize_y = data_model.normalize_y
            self.shape_kernel.normalize_x = True

        self.idx_shape = idx_x + idx_y

        self.idx_continuous = sorted(
            [data_model.inputs.get_keys().index(k) for k in data_model.continuous_keys],
        )

        if len(self.idx_continuous) > 0:
            # bounds = torch.tensor(
            #     data_model.inputs.get_by_keys(data_model.continuous_keys).get_bounds(
            #         specs={},
            #     ),
            # ).to(**tkwargs)
            # norm = Normalize(
            #     indices=self.idx_continuous,
            #     d=len(data_model.inputs.get_keys()),
            #     bounds=bounds,
            # )

            self.transform = None
        else:
            self.transform = None

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        # interpolated y values at unique x locations
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        if self.continuous_kernel is not None:
            continuous_kernel = kernels.map(
                self.continuous_kernel,
                active_dims=self.idx_continuous,
                ard_num_dims=1,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )
            shape_kernel = kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                ard_num_dims=len(self.idx_shape) if self.ard else None,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )
            covar_module = continuous_kernel * shape_kernel
        else:
            shape_kernel = kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                ard_num_dims=len(self.idx_shape) if self.ard else None,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )
            covar_module = shape_kernel

        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=covar_module,
            outcome_transform=(Standardize(m=tY.shape[-1])),
            input_transform=self.transform,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore
        self.model.likelihood.noise_covar.raw_noise_constraint = GreaterThan(5e-4)  # type: ignore

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
