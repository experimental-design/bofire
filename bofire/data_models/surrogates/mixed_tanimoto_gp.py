from typing import Literal, Dict, Union

from pydantic import Field, validator

# from bofire.data_models.enum import MolecularEncodingEnum
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    AnyMolecularKernel,
    AnyKernel,
    HammondDistanceKernel,
    TanimotoKernel,
    MaternKernel,
    ScaleKernel,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.single_task_gp import ScalerEnum
from bofire.data_models.features.molecular import MolecularInput
from bofire.data_models.surrogates.trainable import TrainableSurrogate
from bofire.data_models.molfeatures.api import *


class MixedTanimotoGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    type: Literal["MixedTanimotoGPSurrogate"] = "MixedTanimotoGPSurrogate"

    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )
    # Molecular kernel will only be imposed on fingerprints, fragments, or fingerprintsfragments
    molecular_kernel: AnyMolecularKernel = Field(
        default_factory=lambda: TanimotoKernel(ard=True)
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @validator("input_preprocessing_specs")
    def validate_moleculars(cls, v, values):
        """Checks that at least one of fingerprints, fragments, or fingerprintsfragments features are present."""
        if not any (
            [isinstance(value, Fingerprints)
            or isinstance(value, Fragments)
            or isinstance(value, FingerprintsFragments)
            for value in v.values()
            ]
        ):
            raise ValueError(
                "MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present."
            )
        return v
