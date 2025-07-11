from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import (
    FullyBayesianLinearSingleTaskGP,
    FullyBayesianSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    FullyBayesianSingleTaskGPSurrogate as DataModel,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


_model_mapper = {
    "saas": SaasFullyBayesianSingleTaskGP,
    "linear": FullyBayesianLinearSingleTaskGP,
    "hvarfner": FullyBayesianSingleTaskGP,
}


class FullyBayesianSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.warmup_steps = data_model.warmup_steps
        self.num_samples = data_model.num_samples
        self.thinning = data_model.thinning
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.features_to_warp = data_model.features_to_warp
        self.model_type = data_model.model_type
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[SaasFullyBayesianSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, disable_progbar: bool = True):  # type: ignore
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )
        try:
            self.model = _model_mapper[self.model_type](
                train_X=tX,
                train_Y=tY,
                outcome_transform=(
                    Standardize(m=1)
                    if self.output_scaler == ScalerEnum.STANDARDIZE
                    else None
                ),
                input_transform=scaler,
                use_input_warping=True if len(self.features_to_warp) > 0 else False,
                indices_to_warp=self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, self.features_to_warp
                )
                if len(self.features_to_warp) > 0
                else None,  # type: ignore
            )
        except TypeError:
            # For the current release versions of BoTorch,
            # the `use_input_warping` argument is not available
            # we have to wait for the next release
            self.model = _model_mapper[self.model_type](
                train_X=tX,
                train_Y=tY,
                outcome_transform=(
                    Standardize(m=1)
                    if self.output_scaler == ScalerEnum.STANDARDIZE
                    else None
                ),
                input_transform=scaler,
            )

        fit_fully_bayesian_model_nuts(
            self.model,  # type: ignore
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=disable_progbar,
        )

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore

        preds = posterior.mixture_mean.detach().numpy()
        stds = np.sqrt(posterior.mixture_variance.detach().numpy())
        return preds, stds
