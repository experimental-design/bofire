from typing import Dict, Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    AdditiveMapSaasSingleTaskGPSurrogate as DataModel,
)
from bofire.surrogates.botorch import TrainableBotorchSurrogate


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
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

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        self.model = AdditiveMapSaasSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            num_taus=self.n_taus,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)
