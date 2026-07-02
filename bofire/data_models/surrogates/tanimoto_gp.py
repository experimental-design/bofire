from typing import Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, ScaleKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.molfeatures.api import Fingerprints, Fragments
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TanimotoGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
    )
    tanimoto_calculation_mode: Literal["pre_computed", "on_the_fly"] = "pre_computed"

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
                    for generators in encoding.generators.values()
                    for generator in generators
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
