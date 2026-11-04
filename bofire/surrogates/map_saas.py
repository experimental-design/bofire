from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    EnsembleMapSaasSingleTaskGPSurrogate as DataModel,
)
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate
from bofire.surrogates.utils import scipy_options
from bofire.utils.torch_tools import tkwargs


class EnsembleMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
        scaler: AnyScaler = Normalize(),  # noqa: B008 - copied by make_surrogate
        output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE,
        n_taus: PositiveInt = 4,
        optimizer: LBFGSB = LBFGSB(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create an ensemble MAP-SAAS GP using validated data-model settings.

        Args:
            inputs: Input features.
            outputs: Output features.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            output_scaler: Output scaling method.
            n_taus: Number of SAAS ensemble members.
            optimizer: Settings for fitting the GP hyperparameters.

        Defaults mirror the data model and are copied for each call. Call ``fit`` to train.
        """
        return make_surrogate(cls, DataModel, locals())

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_taus = data_model.n_taus
        self.optimizer = data_model.optimizer
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[EnsembleMapSaasSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        # EnsembleMapSaasSingleTaskGP repeats the data to create a batch dimension
        # The Standardize outcome_transform needs to have the correct batch_shape
        if isinstance(outcome_transform, Standardize):
            outcome_transform = Standardize(
                m=tY.shape[-1],
                batch_shape=torch.Size([self.n_taus]),
            )
        self.model = EnsembleMapSaasSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            num_taus=self.n_taus,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": scipy_options(self.optimizer)},
            max_attempts=self.optimizer.max_attempts,
        )

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            posterior = self.model.posterior(X=X, observation_noise=True)

        preds = posterior.mixture_mean.detach().numpy()
        stds = np.sqrt(posterior.mixture_variance.detach().numpy())
        return preds, stds
