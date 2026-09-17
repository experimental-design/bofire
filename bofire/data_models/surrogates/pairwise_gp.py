from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, RBFKernel, ScaleKernel
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
    """Gaussian process fitted to pairwise preferences rather than measured values.

    Use it when the response can only be judged by comparison — which of two samples
    smells better, looks better, handles better. Instead of experiment values it is
    given winner/loser pairs, and the single output feature it predicts is the latent
    utility that explains them, on an arbitrary scale.
    """

    type: Literal["PairwiseGPSurrogate"] = "PairwiseGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=RBFKernel(
                ard=True,
                lengthscale_prior=PAIRWISEGP_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=PAIRWISEGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=PAIRWISEGP_OUTPUTSCALE_PRIOR(),
            outputscale_constraint=PAIRWISEGP_OUTPUTSCALE_CONSTRAINT(),
        ),
        description="Covariance function of the latent utility, encoding how similar "
        "two candidates' preferences are expected to be.",
    )
    scaler: AnyScaler = Field(
        default_factory=Normalize,
        description="How the inputs are rescaled before fitting. Set to null to leave "
        "them as they are.",
    )
    likelihood: Literal["probit", "logit"] = Field(
        default="probit",
        description="How a difference in latent utility becomes a probability that one "
        "candidate is preferred: probit assumes Gaussian comparison noise, logit "
        "logistic noise, giving the Bradley-Terry model.",
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
