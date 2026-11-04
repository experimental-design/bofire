from typing import Dict, Optional

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.relevance_pursuit import backward_relevance_pursuit
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing_extensions import Self

import bofire.kernels.api as kernels
import bofire.likelihoods.api as likelihoods
import bofire.means.api as means
from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.kernels.api import AnyKernel, RBFKernel
from bofire.data_models.likelihoods.api import (
    HVARFNER_LIKELIHOOD,
    AnyLikelihood,
    SparseOutlierGaussianLikelihood,
)
from bofire.data_models.means.api import AnyMean, ConstantMean
from bofire.data_models.priors.api import HVARFNER_LENGTHSCALE_PRIOR, GreaterThan
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate
from bofire.surrogates.utils import scipy_options


class SingleTaskGPSurrogate(TrainableBotorchSurrogate):
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
        kernel: AnyKernel = RBFKernel(  # noqa: B008 - copied by make_surrogate
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),  # noqa: B008 - copied by make_surrogate
            lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),  # noqa: B008 - copied by make_surrogate
        ),
        mean: AnyMean = ConstantMean(),  # noqa: B008 - copied by make_surrogate
        likelihood: AnyLikelihood = HVARFNER_LIKELIHOOD(),  # noqa: B008 - copied by make_surrogate
        optimizer: LBFGSB = LBFGSB(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create a runtime GP using validated data-model settings.

        Args:
            inputs: Input features.
            outputs: Output features.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            output_scaler: Output scaling method.
            kernel: Covariance kernel.
            mean: Mean function.
            likelihood: Observation noise model, including sparse outlier noise.
            optimizer: Settings for fitting the GP hyperparameters.

        Defaults mirror the data model and are copied for each call. Call ``fit`` to train.
        """
        return make_surrogate(cls, DataModel, locals())

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.mean = data_model.mean
        self.likelihood = data_model.likelihood
        self.optimizer = data_model.optimizer
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        if input_transform is not None:
            n_dim = input_transform(tX).shape[-1]
        else:
            n_dim = tX.shape[-1]

        model_kwargs = {
            "train_X": tX,
            "train_Y": tY,
            "covar_module": kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=self.get_feature_indices,
                inputs=self.inputs,
            ),
            "mean_module": means.map(self.mean, d=n_dim),
            "likelihood": likelihoods.map(self.likelihood, d=n_dim),
            "outcome_transform": outcome_transform,
            "input_transform": input_transform,
        }
        optimizer_kwargs = {"options": scipy_options(self.optimizer)}

        if isinstance(self.likelihood, SparseOutlierGaussianLikelihood):
            self.model = RobustRelevancePursuitSingleTaskGP(
                **model_kwargs,
                convex_parameterization=self.likelihood.convex_parametrization,
                prior_mean_of_support=self.likelihood.prior_mean_of_support,
                cache_model_trace=self.likelihood.cache_model_trace,
            )
            fit_kwargs = {
                "reset_parameters": False,
                "relevance_pursuit_optimizer": backward_relevance_pursuit,
            }
        else:
            self.model = botorch.models.SingleTaskGP(**model_kwargs)
            fit_kwargs = {"max_attempts": self.optimizer.max_attempts}

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs, **fit_kwargs)

    def predict_outliers(
        self, experiments: pd.DataFrame, options: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Fit the model to the experiments and flag the outliers among them.

        Args:
            experiments: The experiments to fit to and assess.
            options: Keyword arguments passed to `fit`.

        Returns:
            The predictions for the experiments, plus a column `{output key}_rho` with
            each experiment's own noise level; non-zero marks an outlier.

        Raises:
            ValueError: If the likelihood is not a `SparseOutlierGaussianLikelihood`.
        """
        if not isinstance(self.likelihood, SparseOutlierGaussianLikelihood):
            raise ValueError(
                "Outliers are predicted with a SparseOutlierGaussianLikelihood only."
            )
        self.fit(experiments=experiments, **(options or {}))
        predictions = self.predict(experiments)
        rhos = self.model.likelihood.noise_covar.rho.cpu().detach().numpy()
        rho_df = pd.DataFrame(
            data=rhos,
            columns=[f"{col}_rho" for col in self.outputs.get_keys()],
        )
        return pd.concat([predictions, rho_df], axis=1)
