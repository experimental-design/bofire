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

# from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.api import RobustSingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


class RobustSingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    """
    Robust Relevance Pursuit Single Task Gaussian Process Surrogate.

    A robust single-task GP that learns a data-point specific noise level and is therefore more robust to outliers.
    See: https://botorch.org/docs/tutorials/relevance_pursuit_robust_regression/
    Paper: https://arxiv.org/pdf/2410.24222

    Attributes:
        prior_mean_of_support (float): The prior mean of the support.
        convex_parametrization (bool): Whether to use convex parametrization of the sparse noise model.
        cache_model_trace (bool): Whether to cache the model trace. This needs no be set to True if you want to view the model trace after optimization.
        lengthscale_constraint (PriorConstraint): Constraint on the lengthscale of the kernel.

    Note:
        The definition of "outliers" depends on the model capacity, so what is an outlier
        with respect to a simple model might not be an outlier with respect to a complex model.
        For this reason, it is necessary to bound the lengthscale of the GP kernel from below.
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
            convex_parameterization=self.convex_parametrization,
            cache_model_trace=self.cache_model_trace,
        )
        if self.prior_mean_of_support is not None:
            self.model.prior_mean_of_support = self.prior_mean_of_support
        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(
            mll,
            reset_parameters=False,
            relevance_pursuit_optimizer=backward_relevance_pursuit,
        )

    def predict_outliers(
        self, experiments: pd.DataFrame, options: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Predict the outliers for the given experiments. This is done by fitting the RobustSingleTaskGPSurrogate model and then using the model to do
        predictions and obtain the data-point specific noise level (rho).
        """
        options = options or {}

        # (re)fit model can only predict rho values for the data the model is fitted on
        self.fit(experiments=experiments, **options)

        # get model predictions, this should do a lot of validation, so we don't need it for the rhos.
        predictions = self.predict(experiments)

        # get the datapoint specific noise level
        rhos = self.model.likelihood.noise_covar.rho.cpu().detach().numpy()  # type: ignore

        # convert rhos to a DataFrame, this loop is not necessary because we only fit on one output, but possibly future proof.
        rho_df = pd.DataFrame(
            data=rhos,
            columns=[f"{col}_rho" for col in self.outputs.get_keys()],
        )

        return pd.concat([predictions, rho_df], axis=1)
