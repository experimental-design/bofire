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
from bofire.data_models.molfeatures.api import *


class MixedTanimotoGPSurrogate(BotorchSurrogate):
    type: Literal["MixedTanimotoGPSurrogate"] = "MixedTanimotoGPSurrogate"

    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: ScaleKernel(base_kernel=MaternKernel(ard=True, nu=2.5))
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: ScaleKernel(base_kernel=HammondDistanceKernel(ard=True))
    )
    # Modify the default kernel for Mordred descriptors
    molecular_kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(base_kernel=TanimotoKernel(ard=True))
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @validator("input_preprocessing_specs")
    def validate_moleculars(cls, v, values):
        """Checks that at least one of fingerprints, fragments, fingerprints_fragments, or Mordred descriptor features are present."""
        if not any (
            [ isinstance(value, MolFeatures)
              for value in v.values()
            ]
        ):
            raise ValueError(
                "MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, fingerprints_fragments, or Mordred descriptor features are present."
            )
        molecular_features = [
            value for value in v.values() if isinstance(value, MolFeatures)
        ]
        if any ([isinstance(feature, MordredDescriptors) for feature in molecular_features]) and \
        any ([isinstance(feature, Fingerprints) or isinstance(feature, Fragments) or isinstance(feature, FingerprintsFragments) for feature in molecular_features]): 
            raise ValueError(
                "Fingerprints, Fragments, or FingerprintsFragments and Mordred Descriptors cannot present simultaneously in MixedTanimotoGPSurrogate."
            )
        return v
