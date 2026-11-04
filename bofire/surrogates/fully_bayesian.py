from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import (
    FullyBayesianLinearSingleTaskGP,
    FullyBayesianSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from typing_extensions import Self

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import (
    FullyBayesianSingleTaskGPSurrogate as DataModel,
)
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate
from bofire.utils.torch_tools import tkwargs


_model_mapper = {
    "saas": SaasFullyBayesianSingleTaskGP,
    "linear": FullyBayesianLinearSingleTaskGP,
    "hvarfner": FullyBayesianSingleTaskGP,
}


class FullyBayesianSingleTaskGPSurrogate(TrainableBotorchSurrogate):
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
        model_type: Literal["linear", "saas", "hvarfner"] = "saas",
        warmup_steps: int = 256,
        num_samples: int = 128,
        thinning: int = 16,
        features_to_warp: list[str] = [],  # noqa: B006 - copied by make_surrogate
    ) -> Self:
        """Create a runtime fully Bayesian GP using validated data-model settings.

        Args:
            inputs: Input features.
            outputs: Output features.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            output_scaler: Output scaling method.
            model_type: Fully Bayesian model family.
            warmup_steps: Number of MCMC warmup steps.
            num_samples: Number of MCMC samples before thinning.
            thinning: Keep every nth MCMC sample.
            features_to_warp: Input feature keys to warp.

        Defaults mirror the data model and are copied for each call. Call ``fit`` to train.
        """
        return make_surrogate(cls, DataModel, locals())

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

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        disable_progbar: bool = True,
    ):
        self.model = _model_mapper[self.model_type](
            train_X=tX,
            train_Y=tY,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            use_input_warping=True if len(self.features_to_warp) > 0 else False,
            indices_to_warp=self.inputs.get_feature_indices(
                self.categorical_encodings, self.features_to_warp
            )
            if len(self.features_to_warp) > 0
            else None,
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
            posterior = self.model.posterior(X=X, observation_noise=True)

        preds = posterior.mixture_mean.detach().numpy()
        stds = np.sqrt(posterior.mixture_variance.detach().numpy())
        return preds, stds
