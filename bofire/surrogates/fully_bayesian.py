from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import Standardize

from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum
from bofire.data_models.surrogates.api import SaasSingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_dim_subsets
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


class SaasSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.warmup_steps = data_model.warmup_steps
        self.num_samples = data_model.num_samples
        self.thinning = data_model.thinning
        self.scaler = data_model.scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[SaasFullyBayesianSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, disable_progbar: bool = True):
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
        # TODO use here the real bounds
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
        # TODO unit it with singletask gp as it is equivalent --> tidyup needed
        # now perform the actual fit

        self.model = SaasFullyBayesianSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=Standardize(m=1),
            input_transform=scaler,
        )
        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=disable_progbar,
        )

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            posterior = self.model.posterior(X=X)  # type: ignore

        preds = posterior.mixture_mean.detach().numpy()
        stds = np.sqrt(posterior.mixture_variance.detach().numpy())
        return preds, stds
