from typing import Literal, Type

from pydantic import Field, validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, ScaleKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
)
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
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
    scaler: ScalerEnum = ScalerEnum.IDENTITY

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    # TanimotoGP will be used when at least one of fingerprints, fragments, or fingerprintsfragments are present
    @validator("input_preprocessing_specs")
    def validate_moleculars(cls, v, values):
        """Checks that at least one of fingerprints, fragments, or fingerprintsfragments features are present."""
        if not any(
            isinstance(value, Fingerprints)
            or isinstance(value, Fragments)
            or isinstance(value, FingerprintsFragments)
            for value in v.values()
        ):
            raise ValueError(
                "TanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
            )
        return v
