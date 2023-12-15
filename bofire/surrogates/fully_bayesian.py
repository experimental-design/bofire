from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import SaasSingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
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
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[SaasFullyBayesianSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, disable_progbar: bool = True):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        self.model = SaasFullyBayesianSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=Standardize(m=1)
            if self.output_scaler == ScalerEnum.STANDARDIZE
            else None,
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
            posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore

        preds = posterior.mixture_mean.detach().numpy()
        stds = np.sqrt(posterior.mixture_variance.detach().numpy())
        return preds, stds
