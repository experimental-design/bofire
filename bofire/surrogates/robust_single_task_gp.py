from typing import Dict, Optional

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll

# from bofire.data_models.molfeatures.api import MolFeatures
from botorch.models.relevance_pursuit import (
    backward_relevance_pursuit,
    forward_relevance_pursuit,  # noqa: F401
)
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


# TODO: Think if we should inherit SingleTaskGPSurrogate or not
# TODO: add to data_model
class RobustSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    """
    Robust Relevance Pursuit Single Task Gaussian Process Surrogate.
    See: https://botorch.org/docs/tutorials/relevance_pursuit_robust_regression/
    Paper: https://arxiv.org/pdf/2410.24222
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.noise_prior = data_model.noise_prior
        self.prior_mean_of_support = data_model.prior_mean_of_support
        self.convex_parametrization = data_model.convex_parametrization
        self.cache_model_trace = data_model.cache_model_trace
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[RobustRelevancePursuitSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        self.model = RobustRelevancePursuitSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(tX.shape[1])),
                ard_num_dims=1,  # this keyword is ignored
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            ),
            outcome_transform=(
                Standardize(m=tY.shape[-1])
                if self.output_scaler == ScalerEnum.STANDARDIZE
                else None
            ),
            input_transform=scaler,
            prior_mean_of_support=self.prior_mean_of_support,
            convex_parameterization=self.convex_parametrization,
            cache_model_trace=self.cache_model_trace,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(
            mll,
            # TODO: with the Decorator to _fit_rrp in botorch robust_relevance_pursuit.py, options and max_attempts do not work
            # TODO: still find workaround for this
            # options=self.training_specs,
            reset_parameters=False,  # question is if we want to be able to set this ourselves or not?
            relevance_pursuit_optimizer=backward_relevance_pursuit,  # same here, do we want to be able to pass this?
            # max_attempts=10,
        )
