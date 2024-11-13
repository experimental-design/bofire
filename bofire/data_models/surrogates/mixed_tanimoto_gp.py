from typing import Literal, Type

from pydantic import Field, validator

# from bofire.data_models.enum import MolecularEncodingEnum
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import (
    AnyCategoricalKernel,
    AnyContinuousKernel,
    AnyMolecularKernel,
    HammingDistanceKernel,
    MaternKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
)
from bofire.data_models.priors.api import (
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MixedTanimotoGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["MixedTanimotoGPSurrogate"] = "MixedTanimotoGPSurrogate"

    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(
            ard=True,
            nu=2.5,
            lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
        )
    )
    categorical_kernel: AnyCategoricalKernel = Field(
        default_factory=lambda: HammingDistanceKernel(ard=True),
    )
    # Molecular kernel will only be imposed on fingerprints, fragments, or fingerprintsfragments
    molecular_kernel: AnyMolecularKernel = Field(
        default_factory=lambda: TanimotoKernel(ard=True),
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

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
                "MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
            )
        return v
