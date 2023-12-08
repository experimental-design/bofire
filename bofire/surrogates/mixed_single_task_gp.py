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
from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum
from bofire.data_models.surrogates.api import MixedSingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


class MixedSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.continuous_kernel = data_model.continuous_kernel
        self.categorical_kernel = data_model.categorical_kernel
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

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        features2idx, _ = self.inputs._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]

        ord_dims = []
        for feat in self.inputs.get():
            if feat.key not in non_numerical_features:
                ord_dims += features2idx[feat.key]

        # these are the categorical dimesions after applying the OneHotToNumeric transform
        cat_dims = list(
            range(len(ord_dims), len(ord_dims) + len(non_numerical_features))
        )
        # these are the categorical features within the the OneHotToNumeric transform
        categorical_features = {
            features2idx[feat][0]: len(features2idx[feat])
            for feat in non_numerical_features
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
            cont_kernel_factory=partial(kernels.map, data_model=self.continuous_kernel),
            outcome_transform=Standardize(m=tY.shape[-1])
            if self.output_scaler == ScalerEnum.STANDARDIZE
            else None,
            input_transform=tf,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)
