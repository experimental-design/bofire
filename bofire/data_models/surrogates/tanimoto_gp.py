from typing import Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.descriptor_generators.api import Fingerprints, Fragments
from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, ScaleKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.trainable_botorch import (
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class TanimotoGPSurrogate(TrainableBotorchSurrogate):
    """Gaussian process over molecules, comparing them by fingerprint overlap.

    Requires at least one input encoded into fingerprints or fragments.
    """

    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ),
        description=KERNEL_DESCRIPTION,
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: THREESIX_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION,
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
        description=NOISE_CONSTRAINT_DESCRIPTION,
    )
    tanimoto_calculation_mode: Literal["pre_computed", "on_the_fly"] = Field(
        default="pre_computed",
        description="Whether to compute the pairwise molecular similarities once up "
        "front or on demand. Precomputing is faster to fit but holds a matrix "
        "quadratic in the number of distinct molecules.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_moleculars(self):
        """Checks that at least one fingerprint/fragment descriptor encoding is present."""

        def _is_tanimoto_encoding(encoding) -> bool:
            # a DescriptorEncoding with no static columns whose generators are all
            # fingerprints/fragments produces the binary space Tanimoto needs.
            return (
                isinstance(encoding, DescriptorEncoding)
                and not encoding.columns
                and bool(encoding.generators)
                and all(
                    isinstance(generator, (Fingerprints, Fragments))
                    for generator in encoding.generators
                )
            )

        if not any(
            _is_tanimoto_encoding(value)
            for value in self.categorical_encodings.values()
        ):
            raise ValueError(
                "TanimotoGPSurrogate can only be used if at least one fingerprint or "
                "fragment descriptor encoding (no static columns) is present.",
            )
        return self
