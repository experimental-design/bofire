import itertools
from typing import List, Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AdditiveMapSaasKernel,
    AnyKernel,
    HammingDistanceKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import (
    HVARFNER_LIKELIHOOD,
    MBO_LIKELIHOOD,
    THREESIX_LIKELIHOOD,
    AnyLikelihood,
    SparseOutlierGaussianLikelihood,
)
from bofire.data_models.means.api import AnyMean, ConstantMean
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_SCALE_PRIOR,
    GreaterThan,
    NormalPrior,
)
from bofire.data_models.surrogates.optimizer import LBFGSBFitted
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


# base kernels tried by `SingleTaskGPSurrogate.options`, built from ARD and a
# lengthscale prior
_BASE_KERNELS = [
    lambda ard, prior: RBFKernel(ard=ard, lengthscale_prior=prior),
    lambda ard, prior: MaternKernel(nu=1.5, ard=ard, lengthscale_prior=prior),
    lambda ard, prior: MaternKernel(nu=2.5, ard=ard, lengthscale_prior=prior),
]

# (likelihood, lengthscale prior, outputscale prior) tried by
# `SingleTaskGPSurrogate.options`
_PRIOR_SETS = [
    (MBO_LIKELIHOOD, MBO_LENGTHSCALE_PRIOR, MBO_OUTPUTSCALE_PRIOR),
    (THREESIX_LIKELIHOOD, THREESIX_LENGTHSCALE_PRIOR, THREESIX_SCALE_PRIOR),
    (HVARFNER_LIKELIHOOD, HVARFNER_LENGTHSCALE_PRIOR, THREESIX_SCALE_PRIOR),
]


def _copy(engineered_features: Optional[EngineeredFeatures]) -> EngineeredFeatures:
    """A copy of its own for every surrogate a factory builds; none if not given."""
    if engineered_features is None:
        return EngineeredFeatures()
    return engineered_features.model_copy(deep=True)


class SingleTaskGPSurrogate(TrainableBotorchSurrogate, LBFGSBFitted):
    type: Literal["SingleTaskGPSurrogate"] = "SingleTaskGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),
        )
    )
    mean: AnyMean = Field(
        default=ConstantMean(),
        description="Prior mean function, what the model predicts far from any "
        "observation.",
    )
    likelihood: AnyLikelihood = Field(
        default=HVARFNER_LIKELIHOOD(),
        description="How observations scatter around the response, i.e. the model of "
        "the measurement noise.",
    )

    @staticmethod
    def options(
        inputs: Inputs,
        outputs: Outputs,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> List["SingleTaskGPSurrogate"]:
        """Single-task GPs that differ in kernel and priors, as options to choose from.

        One for each combination of base kernel (RBF, Matern 1.5, Matern 2.5), set of
        noise, lengthscale and outputscale priors (MBO, THREESIX, HVARFNER), with and
        without a scale kernel, and with and without ARD: 36 in all. Every other field
        keeps its default.

        Args:
            inputs: The input features of every option.
            outputs: The output feature of every option.
            engineered_features: Quantities derived from the inputs that every option
                sees as well; the kernel works on them too. If not provided, none.

        Returns:
            The options, each with its own kernel and likelihood.
        """
        options = []
        for base_kernel, priors, scale, ard in itertools.product(
            _BASE_KERNELS, _PRIOR_SETS, [True, False], [True, False]
        ):
            likelihood, lengthscale_prior, outputscale_prior = priors
            kernel = base_kernel(ard, lengthscale_prior())
            if scale:
                kernel = ScaleKernel(
                    base_kernel=kernel, outputscale_prior=outputscale_prior()
                )
            options.append(
                SingleTaskGPSurrogate(
                    inputs=inputs,
                    outputs=outputs,
                    engineered_features=_copy(engineered_features),
                    kernel=kernel,
                    likelihood=likelihood(),
                )
            )
        return options

    @staticmethod
    def linear(
        inputs: Inputs,
        outputs: Outputs,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> "SingleTaskGPSurrogate":
        """Single-task GP whose response is linear in the inputs.

        Equivalent to Bayesian linear regression. The noise has a THREESIX prior with a
        lower bound of 1e-4; every other field keeps its default.

        Args:
            inputs: The input features.
            outputs: The output feature.
            engineered_features: Quantities derived from the inputs that the model
                sees as well; the kernel works on them too. If not provided, none.

        Returns:
            The surrogate, with a linear kernel.
        """
        return SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            engineered_features=_copy(engineered_features),
            kernel=LinearKernel(),
            likelihood=THREESIX_LIKELIHOOD(),
        )

    @staticmethod
    def polynomial(
        inputs: Inputs,
        outputs: Outputs,
        power: int = 2,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> "SingleTaskGPSurrogate":
        """Single-task GP whose response is a polynomial in the inputs.

        The noise has a THREESIX prior with a lower bound of 1e-4; every other field
        keeps its default.

        Args:
            inputs: The input features.
            outputs: The output feature.
            engineered_features: Quantities derived from the inputs that the model
                sees as well; the kernel works on them too. If not provided, none.
            power: Degree of the polynomial.

        Returns:
            The surrogate, with a polynomial kernel of the given power.
        """
        return SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            engineered_features=_copy(engineered_features),
            kernel=PolynomialKernel(power=power),
            likelihood=THREESIX_LIKELIHOOD(),
        )

    @staticmethod
    def additive_map_saas(
        inputs: Inputs,
        outputs: Outputs,
        n_taus: int = 4,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> "SingleTaskGPSurrogate":
        """Single-task GP for many inputs of which only a few are expected to matter.

        The maximum-a-posteriori approximation of the fully Bayesian SAAS model, as
        BoTorch's `AdditiveMapSaasSingleTaskGP` builds it: an additive sparse kernel, a
        constant mean with a standard normal prior bounded to [-10, 10], and the default
        likelihood.

        Args:
            inputs: The input features.
            outputs: The output feature.
            engineered_features: Quantities derived from the inputs that the model
                sees as well; the kernel works on them too. If not provided, none.
            n_taus: Number of sparse kernels summed up, each at its own sparsity level.

        Returns:
            The surrogate, with an additive MAP-SAAS kernel.
        """
        return SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            engineered_features=_copy(engineered_features),
            kernel=AdditiveMapSaasKernel(n_taus=n_taus),
            mean=ConstantMean(
                prior=NormalPrior(loc=0.0, scale=1.0), bounds=(-10.0, 10.0)
            ),
        )

    @staticmethod
    def mixed(
        inputs: Inputs,
        outputs: Outputs,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> "SingleTaskGPSurrogate":
        """Single-task GP for continuous and categorical inputs, as BoTorch's
        `MixedSingleTaskGP` builds it.

        Categorical inputs without descriptor data are encoded as ordinal codes and
        compared by a Hamming kernel; all other inputs and the engineered features by
        an RBF kernel with the HVARFNER prior. The kernel is the sum of the two plus
        their product, each scaled; without other inputs, the scaled Hamming kernel.

        Args:
            inputs: The input features, at least one of them categorical without
                descriptor data.
            outputs: The output feature.
            engineered_features: Quantities derived from the inputs that the model
                sees as well; the RBF kernel works on them. If not provided, none.

        Returns:
            The surrogate.

        Raises:
            ValueError: If no categorical input lacks descriptor data, or if an
                engineered feature replaces the features it is computed from.
        """
        engineered_features = _copy(engineered_features)
        if any(not feat.keep_features for feat in engineered_features.get()):
            raise ValueError(
                "The mixed GP needs every engineered feature to keep the features it "
                "is computed from (keep_features=True)."
            )
        categorical = [
            feat.key
            for feat in inputs.get(CategoricalInput, exact=False)
            if feat.descriptors is None
        ]
        if not categorical:
            raise ValueError(
                "The mixed GP needs at least one categorical input without "
                "descriptor data."
            )
        continuous = [
            key for key in inputs.get_keys() if key not in categorical
        ] + engineered_features.get_keys()

        def continuous_kernel() -> RBFKernel:
            return RBFKernel(
                ard=True,
                lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),
                features=continuous,
            )

        def categorical_kernel() -> HammingDistanceKernel:
            return HammingDistanceKernel(ard=True, features=categorical)

        if continuous:
            kernel = AdditiveKernel(
                kernels=[
                    ScaleKernel(
                        base_kernel=AdditiveKernel(
                            kernels=[
                                continuous_kernel(),
                                ScaleKernel(base_kernel=categorical_kernel()),
                            ]
                        )
                    ),
                    ScaleKernel(
                        base_kernel=MultiplicativeKernel(
                            kernels=[continuous_kernel(), categorical_kernel()]
                        )
                    ),
                ]
            )
        else:
            kernel = ScaleKernel(base_kernel=categorical_kernel())
        return SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            engineered_features=engineered_features,
            categorical_encodings={key: OrdinalEncoding() for key in categorical},
            kernel=kernel,
        )

    @staticmethod
    def robust(
        inputs: Inputs,
        outputs: Outputs,
        *,
        engineered_features: Optional[EngineeredFeatures] = None,
    ) -> "SingleTaskGPSurrogate":
        """Single-task GP that detects outliers and discounts them: BoTorch's
        `RobustRelevancePursuitSingleTaskGP`, with the lengthscales bounded at 0.05
        instead of 0.025.

        The likelihood adds an extra noise variance for observations treated as
        outliers to the HVARFNER noise model. The kernel is an RBF kernel with the
        HVARFNER prior whose lengthscales are at least 0.05, so that the model cannot
        simply explain the outliers.

        Args:
            inputs: The input features.
            outputs: The output feature, exactly one.
            engineered_features: Quantities derived from the inputs that the model
                sees as well; the kernel works on them too. If not provided, none.

        Returns:
            The surrogate, with a `SparseOutlierGaussianLikelihood`.
        """
        return SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            engineered_features=_copy(engineered_features),
            kernel=RBFKernel(
                ard=True,
                lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            ),
            likelihood=SparseOutlierGaussianLikelihood(
                noise_prior=HVARFNER_NOISE_PRIOR()
            ),
        )

    @model_validator(mode="after")
    def validate_sparse_outlier_likelihood(self):
        """Outlier noise levels are fitted for one output only."""
        if (
            isinstance(self.likelihood, SparseOutlierGaussianLikelihood)
            and len(self.outputs) > 1
        ):
            raise ValueError(
                "A SparseOutlierGaussianLikelihood supports one output only."
            )
        return self

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
