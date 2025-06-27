from typing import Dict, Optional

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    AdditiveMapSaasSingleTaskGPSurrogate as DataModel,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


class AdditiveMapSaasSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_taus = data_model.n_taus
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[AdditiveMapSaasSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, disable_progbar: bool = True):  # type: ignore
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        self.model = AdditiveMapSaasSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=(
                Standardize(m=1)
                if self.output_scaler == ScalerEnum.STANDARDIZE
                else None
            ),
            input_transform=scaler,
            num_taus=self.n_taus,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
