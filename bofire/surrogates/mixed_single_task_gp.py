from functools import partial
from typing import Dict, Optional

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import ChainedInputTransform, OneHotToNumeric
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import MixedSingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import (
    get_categorical_feature_keys,
    get_continuous_feature_keys,
    get_scaler,
)
from bofire.utils.torch_tools import tkwargs


class MixedSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.continuous_kernel = data_model.continuous_kernel
        self.categorical_kernel = data_model.categorical_kernel
        self.noise_prior = data_model.noise_prior
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MixedSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    # features2idx: Optional[Dict] = None
    # non_numerical_features: Optional[List] = None
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        continuous_feature_keys = get_continuous_feature_keys(
            self.inputs,
            self.input_preprocessing_specs,
        )
        ord_dims = self.inputs.get_feature_indices(
            self.input_preprocessing_specs,
            continuous_feature_keys,
        )

        categorical_feature_keys = get_categorical_feature_keys(
            self.input_preprocessing_specs,
        )
        # these are the categorical dimensions after applying the OneHotToNumeric transform
        cat_dims = list(
            range(len(ord_dims), len(ord_dims) + len(categorical_feature_keys)),
        )

        features2idx, _ = self.inputs._get_transform_info(
            self.input_preprocessing_specs,
        )

        # these are the categorical features within the the OneHotToNumeric transform
        categorical_features = {
            features2idx[feat][0]: len(features2idx[feat])
            for feat in categorical_feature_keys
        }

        o2n = OneHotToNumeric(
            dim=tX.shape[1],
            categorical_features=categorical_features,
            transform_on_train=False,
        )
        tf = ChainedInputTransform(tf1=scaler, tf2=o2n) if scaler is not None else o2n

        # fit the model
        self.model = botorch.models.MixedSingleTaskGP(
            train_X=o2n.transform(tX),
            train_Y=tY,
            cat_dims=cat_dims,
            # cont_kernel_factory=self.continuous_kernel.to_gpytorch,
            cont_kernel_factory=partial(
                kernels.map,
                data_model=self.continuous_kernel,
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            ),
            outcome_transform=(
                Standardize(m=tY.shape[-1])
                if self.output_scaler == ScalerEnum.STANDARDIZE
                else None
            ),
            input_transform=tf,
        )
        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)
