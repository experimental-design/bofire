from typing import Literal, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform
from typing_extensions import Self

import bofire.kernels.api as kernels
from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.kernels.api import AnyKernel, RBFKernel, ScaleKernel
from bofire.data_models.priors.api import (
    PAIRWISEGP_LENGTHSCALE_CONSTRAINT,
    PAIRWISEGP_LENGTHSCALE_PRIOR,
    PAIRWISEGP_OUTPUTSCALE_CONSTRAINT,
    PAIRWISEGP_OUTPUTSCALE_PRIOR,
)
from bofire.data_models.surrogates.api import PairwiseGPSurrogate as DataModel
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.pairwise_trainable import PairwiseTrainableSurrogate
from bofire.surrogates.surrogate import make_surrogate
from bofire.surrogates.utils import scipy_options


# maps the serializable likelihood name to the BoTorch PairwiseLikelihood class
PAIRWISE_LIKELIHOODS = {
    "probit": PairwiseProbitLikelihood,
    "logit": PairwiseLogitLikelihood,
}


class PairwiseGPSurrogate(BotorchSurrogate, PairwiseTrainableSurrogate):
    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
        scaler: AnyScaler = Normalize(),  # noqa: B008 - copied by make_surrogate
        kernel: AnyKernel = ScaleKernel(  # noqa: B008 - copied by make_surrogate
            base_kernel=RBFKernel(  # noqa: B008 - copied by make_surrogate
                ard=True,
                lengthscale_prior=PAIRWISEGP_LENGTHSCALE_PRIOR(),  # noqa: B008 - copied by make_surrogate
                lengthscale_constraint=PAIRWISEGP_LENGTHSCALE_CONSTRAINT(),  # noqa: B008 - copied by make_surrogate
            ),
            outputscale_prior=PAIRWISEGP_OUTPUTSCALE_PRIOR(),  # noqa: B008 - copied by make_surrogate
            outputscale_constraint=PAIRWISEGP_OUTPUTSCALE_CONSTRAINT(),  # noqa: B008 - copied by make_surrogate
        ),
        likelihood: Literal["probit", "logit"] = "probit",
        optimizer: LBFGSB = LBFGSB(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create a pairwise GP using validated data-model settings.

        Args:
            inputs: Input features.
            outputs: One continuous output representing latent utility.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            kernel: Scaled covariance kernel.
            likelihood: Probit or logit comparison likelihood.
            optimizer: Settings for fitting the GP hyperparameters.

        Defaults mirror the data model and are copied for each call.
        Call ``fit`` with experiments and preferences to train.
        """
        return make_surrogate(cls, DataModel, locals())

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.optimizer = data_model.optimizer
        self.scaler = data_model.scaler
        self.likelihood = data_model.likelihood
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.PairwiseGP] = None

    def _fit_pairwise(
        self,
        datapoints: torch.Tensor,
        comparisons: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        **kwargs,
    ):
        if input_transform is not None:
            n_dim = input_transform(datapoints).shape[-1]
        else:
            n_dim = datapoints.shape[-1]

        self.model = botorch.models.PairwiseGP(
            datapoints=datapoints,
            comparisons=comparisons,
            likelihood=PAIRWISE_LIKELIHOODS[self.likelihood](),
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=self.get_feature_indices,
                inputs=self.inputs,
            ),
            input_transform=input_transform,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=self.model.likelihood,
            model=self.model,
        )
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": scipy_options(self.optimizer)},
            max_attempts=self.optimizer.max_attempts,
        )

    # `_predict` is inherited from BotorchSurrogate: it calls
    # `posterior(X, observation_noise=True)`, and PairwiseGP ignores
    # `observation_noise` (verified in scripts/pairwise_gp_checks.py).
