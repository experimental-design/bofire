from typing import Dict, List, Literal, Optional

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputStandardize,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import Field

from bofire.any.kernel import AnyKernel
from bofire.models.gps.kernels import (
    CategoricalKernel,
    ContinuousKernel,
    HammondDistanceKernel,
    MaternKernel,
    ScaleKernel,
)
from bofire.models.gps.priors import botorch_lengthcale_prior, botorch_scale_prior
from bofire.models.model import TrainableModel
from bofire.models.torch_models import BotorchModel
from bofire.utils.enum import CategoricalEncodingEnum, OutputFilteringEnum, ScalerEnum
from bofire.utils.torch_tools import OneHotToNumeric, tkwargs


def get_dim_subsets(d: int, active_dims: List[int], cat_dims: List[int]):
    def check_indices(d, indices):
        if len(set(indices)) != len(indices):
            raise ValueError("Elements of `indices` list must be unique!")
        if any([i > d - 1 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        if len(indices) > d:
            raise ValueError("Can provide at most `d` indices!")
        if any([i < 0 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        return indices

    if len(active_dims) == 0:
        raise ValueError("At least one active dim has to be provided!")

    # check validity of indices
    active_dims = check_indices(d, active_dims)
    cat_dims = check_indices(d, cat_dims)

    # compute subsets
    ord_dims = sorted(set(range(d)) - set(cat_dims))
    ord_active_dims = sorted(
        set(active_dims) - set(cat_dims)
    )  # includes also descriptors
    cat_active_dims = sorted([i for i in cat_dims if i in active_dims])
    return ord_dims, ord_active_dims, cat_active_dims


class SingleTaskGPModel(BotorchModel, TrainableModel):
    type: Literal["SingleTaskGPModel"] = "SingleTaskGPModel"
    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=botorch_lengthcale_prior()
            ),
            outputscale_prior=botorch_scale_prior(),
        )
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}  # only relevant for training

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]
        # transform X
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        # transform from pandas to torch
        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        if tX.dim() == 2:
            batch_shape = torch.Size()
        else:
            batch_shape = torch.Size([tX.shape[0]])

        d = tX.shape[-1]

        cat_dims = []
        for feat in non_numerical_features:
            cat_dims += features2idx[feat]

        ord_dims, _, _ = get_dim_subsets(
            d=d, active_dims=list(range(d)), cat_dims=cat_dims
        )
        # first get the scaler
        # TODO use here the correct bounds
        if self.scaler == ScalerEnum.NORMALIZE:
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )

            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                batch_shape=batch_shape,
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims if len(ord_dims) != d else None,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError("Scaler enum not known.")

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=self.kernel.to_gpytorch(
                batch_shape=batch_shape, active_dims=list(range(d)), ard_num_dims=1
            ),
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=scaler,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)


class MixedSingleTaskGPModel(BotorchModel, TrainableModel):
    type: Literal["MixedSingleTaskGPModel"] = "MixedSingleTaskGPModel"
    continuous_kernel: ContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: CategoricalKernel = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    model: Optional[botorch.models.MixedSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    # features2idx: Optional[Dict] = None
    # non_numerical_features: Optional[List] = None
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]
        # transform X
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        # transform from pandas to torch
        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        if tX.dim() == 2:
            batch_shape = torch.Size()
        else:
            batch_shape = torch.Size([tX.shape[0]])

        # get indices of the continuous and categorical dims
        d = tX.shape[-1]

        ord_dims = []
        for feat in self.input_features.get():
            if feat.key not in non_numerical_features:
                ord_dims += features2idx[feat.key]
        cat_dims = list(
            range(len(ord_dims), len(ord_dims) + len(non_numerical_features))
        )
        categorical_features = {
            features2idx[feat][0]: len(features2idx[feat])
            for feat in non_numerical_features
        }

        # first get the scaler
        if self.scaler == ScalerEnum.NORMALIZE:
            # TODO: take the real bounds here
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )
            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                indices=ord_dims,
                batch_shape=batch_shape,
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError("Scaler enum not known.")

        o2n = OneHotToNumeric(
            dim=d, categorical_features=categorical_features, transform_on_train=False
        )
        tf = ChainedInputTransform(tf1=scaler, tf2=o2n)

        # fit the model
        self.model = botorch.models.MixedSingleTaskGP(
            train_X=o2n.transform(tX),
            train_Y=tY,
            cat_dims=cat_dims,
            cont_kernel_factory=self.continuous_kernel.to_gpytorch,
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=tf,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)
