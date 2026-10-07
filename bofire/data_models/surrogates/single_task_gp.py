import itertools
from typing import List, Literal, Type

from pydantic import Field

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import (
    AnyKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import AnyLikelihood, GaussianLikelihood
from bofire.data_models.means.api import AnyMean, ConstantMean
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


# base kernels tried by `SingleTaskGPSurrogate.options`, built from ARD and a
# lengthscale prior
_BASE_KERNELS = [
    lambda ard, prior: RBFKernel(ard=ard, lengthscale_prior=prior),
    lambda ard, prior: MaternKernel(nu=1.5, ard=ard, lengthscale_prior=prior),
    lambda ard, prior: MaternKernel(nu=2.5, ard=ard, lengthscale_prior=prior),
]

# (noise, lengthscale, outputscale) priors tried by `SingleTaskGPSurrogate.options`
_PRIOR_SETS = [
    (MBO_NOISE_PRIOR, MBO_LENGTHSCALE_PRIOR, MBO_OUTPUTSCALE_PRIOR),
    (THREESIX_NOISE_PRIOR, THREESIX_LENGTHSCALE_PRIOR, THREESIX_SCALE_PRIOR),
    (HVARFNER_NOISE_PRIOR, HVARFNER_LENGTHSCALE_PRIOR, THREESIX_SCALE_PRIOR),
]


class SingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["SingleTaskGPSurrogate"] = "SingleTaskGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
        )
    )
    mean: AnyMean = Field(
        default=ConstantMean(),
        description="Prior mean function, what the model predicts far from any "
        "observation.",
    )
    likelihood: AnyLikelihood = Field(
        default=GaussianLikelihood(),
        description="How observations scatter around the response, i.e. the model of "
        "the measurement noise.",
    )

    @staticmethod
    def options(inputs: Inputs, outputs: Outputs) -> List["SingleTaskGPSurrogate"]:
        """Single-task GPs that differ in kernel and priors, as candidates to choose from.

        One for each combination of base kernel (RBF, Matern 1.5, Matern 2.5), set of
        noise, lengthscale and outputscale priors (MBO, THREESIX, HVARFNER), with and
        without a scale kernel, and with and without ARD: 36 in all. Every other field
        keeps its default.

        Args:
            inputs: The input features of every option.
            outputs: The output feature of every option.

        Returns:
            The options, each with its own kernel and likelihood.
        """
        options = []
        for base_kernel, priors, scale, ard in itertools.product(
            _BASE_KERNELS, _PRIOR_SETS, [True, False], [True, False]
        ):
            noise_prior, lengthscale_prior, outputscale_prior = priors
            kernel = base_kernel(ard, lengthscale_prior())
            if scale:
                kernel = ScaleKernel(
                    base_kernel=kernel, outputscale_prior=outputscale_prior()
                )
            options.append(
                SingleTaskGPSurrogate(
                    inputs=inputs,
                    outputs=outputs,
                    kernel=kernel,
                    likelihood=GaussianLikelihood(noise_prior=noise_prior()),
                )
            )
        return options

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
