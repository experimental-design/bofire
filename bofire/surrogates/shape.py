from typing import Dict, Optional

import botorch
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
from bofire.data_models.features.api import CloneFeature, InterpolateFeature
from bofire.data_models.kernels.api import (
    ExactWassersteinKernel as ExactWassersteinKernelData,
)
from bofire.data_models.surrogates.api import PiecewiseLinearGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.engineered_features import map as map_feature
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


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

        idx_x = [data_model.inputs.get_keys().index(k) for k in data_model.x_keys]
        idx_y = [data_model.inputs.get_keys().index(k) for k in data_model.y_keys]
        self.idx_x = idx_x
        self.idx_y = idx_y

        self.prepend_x = torch.tensor(data_model.prepend_x).to(**tkwargs)
        self.prepend_y = torch.tensor(data_model.prepend_y).to(**tkwargs)
        self.append_x = torch.tensor(data_model.append_x).to(**tkwargs)
        self.append_y = torch.tensor(data_model.append_y).to(**tkwargs)
        self.normalize_y = torch.tensor(data_model.normalize_y).to(**tkwargs)

        interpolate_feature = InterpolateFeature(
            key="__interpolated_shape__",
            features=data_model.x_keys + data_model.y_keys,
            x_keys=data_model.x_keys,
            y_keys=data_model.y_keys,
            interpolation_range=data_model.interpolation_range,
            n_interpolation_points=data_model.n_interpolation_points,
            prepend_x=data_model.prepend_x,
            prepend_y=data_model.prepend_y,
            append_x=data_model.append_x,
            append_y=data_model.append_y,
            normalize_y=data_model.normalize_y,
            normalize_x=True,
        )
        inter = map_feature(
            data_model=interpolate_feature,
            inputs=data_model.inputs,
            transform_specs=data_model.input_preprocessing_specs,
        )

        self.idx_shape = list(range(data_model.n_interpolation_points))

        self.idx_continuous = sorted(
            [
                data_model.inputs.get_keys().index(k)
                + data_model.n_interpolation_points
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
                d=len(data_model.inputs.get_keys()) + data_model.n_interpolation_points,
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
            covar_module = kernels.map(
                self.continuous_kernel,
                active_dims=self.idx_continuous,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            ) * kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )
        else:
            covar_module = kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
                batch_shape=torch.Size(),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            )

        # if self.saas and self.continuous_kernel is None:
        #     self.model = botorch.models.map_saas.AdditiveMapSaasSingleTaskGP(
        #         train_X=tX,
        #         train_Y=tY,
        #         outcome_transform=(Standardize(m=tY.shape[-1])),
        #         input_transform=self.transform,
        #     )  # type: ignore
        # elif self.saas and self.continuous_kernel is not None:
        #     raise ValueError(
        #         "SAAS is only implemented for shape-only models without continuous kernel."
        #     )
        # else:
        self.model = botorch.models.SingleTaskGP(  # type: ignore
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

        idx_continuous_original = sorted(
            [data_model.inputs.get_keys().index(k) for k in data_model.continuous_keys],
        )
        self.idx_continuous = idx_continuous_original
        if len(idx_continuous_original) > 0:
            clone_feature = CloneFeature(
                key="__shape_continuous_clones__",
                features=data_model.continuous_keys,
            )
            clone_transform = map_feature(
                data_model=clone_feature,
                inputs=data_model.inputs,
                transform_specs=data_model.input_preprocessing_specs,
            )
            d = len(data_model.inputs.get_keys()) + len(idx_continuous_original)
            self.idx_continuous = list(
                range(len(data_model.inputs.get_keys()), d),
            )
            self.idx_shape_clones = self.idx_continuous
            continuous_bounds = torch.tensor(
                data_model.inputs.get_by_keys(data_model.continuous_keys).get_bounds(
                    specs={},
                ),
            ).to(**tkwargs)
            continuous_norm = Normalize(
                indices=self.idx_continuous,
                d=d,
                bounds=continuous_bounds,
            )
            self._continuous_feature_to_idx = dict(
                zip(data_model.continuous_keys, self.idx_continuous),
            )
            self.transform = ChainedInputTransform(
                tf1=clone_transform,
                tf2=continuous_norm,
            )
        else:
            self.idx_shape_clones = []
            self._continuous_feature_to_idx = {}
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

            def _continuous_features_to_idx(features: list[str]) -> list[int]:
                return sorted(
                    self._continuous_feature_to_idx[feat] for feat in features
                )

            continuous_kernel = kernels.map(
                self.continuous_kernel,
                active_dims=self.idx_continuous,
                batch_shape=torch.Size(),
                features_to_idx_mapper=_continuous_features_to_idx,
            )
            shape_kernel = kernels.map(
                self.shape_kernel,
                active_dims=self.idx_shape,
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
