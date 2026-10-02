from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, RBFKernel, ScaleKernel
from bofire.data_models.likelihoods.api import (
    AnyPairwiseLikelihood,
    PairwiseProbitLikelihood,
)
from bofire.data_models.priors.api import (
    PAIRWISEGP_LENGTHSCALE_CONSTRAINT,
    PAIRWISEGP_LENGTHSCALE_PRIOR,
    PAIRWISEGP_OUTPUTSCALE_CONSTRAINT,
    PAIRWISEGP_OUTPUTSCALE_PRIOR,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class PairwiseGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    """Pairwise Gaussian Process surrogate built on top of BoTorch's PairwiseGP.

    Fits a latent utility function from binary winner/loser pair labels. The
    `preferences` DataFrame references rows of the standard BoFire `experiments`
    DataFrame by `labcode`; the single output feature represents the latent
    utility inferred from those comparisons.

    Attributes:
        likelihood: How differences in latent utility determine preference
            probabilities: PairwiseProbitLikelihood (BoTorch's default, assuming
            Gaussian noise) or PairwiseLogitLikelihood (the Bradley-Terry model).
    """

    type: Literal["PairwiseGPSurrogate"] = "PairwiseGPSurrogate"
    hyperconfig: None = Field(
        default=None,
        description="PairwiseGP hyperparameter optimization is not supported.",
    )

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=RBFKernel(
                ard=True,
                lengthscale_prior=PAIRWISEGP_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=PAIRWISEGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=PAIRWISEGP_OUTPUTSCALE_PRIOR(),
            outputscale_constraint=PAIRWISEGP_OUTPUTSCALE_CONSTRAINT(),
        )
    )
    scaler: AnyScaler = Field(default_factory=Normalize)
    likelihood: AnyPairwiseLikelihood = Field(
        default_factory=PairwiseProbitLikelihood,
        description="How latent-utility differences determine the probability of "
        "preferring one alternative over another.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_single_output(self):
        if len(self.outputs) != 1:
            raise ValueError(
                "PairwiseGPSurrogate supports exactly one output (the latent utility)."
            )
        return self

    @model_validator(mode="after")
    def validate_scalekernel(self):
        if not isinstance(self.kernel, ScaleKernel):
            raise ValueError(
                "PairwiseGPSurrogate.kernel must be a ScaleKernel "
                "(BoTorch's PairwiseGP requires the covariance module to be a ScaleKernel)."
            )
        return self
