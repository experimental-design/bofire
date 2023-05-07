from typing import Literal, Dict, Union

from pydantic import Field, validator

from bofire.data_models.enum import MolecularEncodingEnum
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    HammingDistanceKernel,
    TanimotoKernel,
    MaternKernel,
    ScaleKernel,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.single_task_gp import ScalerEnum
from bofire.data_models.features.molecular import MolecularInput


class MixedTanimotoGPSurrogate(BotorchSurrogate):
    type: Literal["MixedTanimotoGPSurrogate"] = "MixedTanimotoGPSurrogate"

    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: ScaleKernel(base_kernel=MaternKernel(ard=True, nu=2.5))
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: ScaleKernel(base_kernel=HammingDistanceKernel(ard=True))
    )
    molecular_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: ScaleKernel(base_kernel=TanimotoKernel())
    )
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @validator("input_preprocessing_specs")
    def validate_categoricals(cls, v, values):
        """Checks that at least one of fingerprints, fragments or fingerprints_fragments features are present."""
        if (
            MolecularEncodingEnum.FINGERPRINTS not in v.values()
            and MolecularEncodingEnum.FRAGMENTS not in v.values()
            and MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS not in v.values()
        ):
            raise ValueError(
                "MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments or fingerprints_fragments features are present."
            )
        if not any([not isinstance(x, MolecularInput) for x in values["inputs"].get()]):
            if MolecularEncodingEnum.MOL_DESCRIPTOR not in v.values():
                raise ValueError(
                    "Did not find least one continuous, categorical, or molecular descriptors features. MixedTanimotoGPSurrogate is designed to be used with any of these in combination with molecualr fingerprints, fragments or fingerprints_fragments features."
                )
        return v
