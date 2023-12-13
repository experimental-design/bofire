from typing import Literal

from pydantic import Field, validator

# from bofire.data_models.enum import MolecularEncodingEnum
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    AnyMolecularKernel,
    HammondDistanceKernel,
    MaternKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
)
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
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
            lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR(),
        )
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )
    # Molecular kernel will only be imposed on fingerprints, fragments, or fingerprintsfragments
    molecular_kernel: AnyMolecularKernel = Field(
        default_factory=lambda: TanimotoKernel(ard=True)
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())

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
                "MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present."
            )
        return v
