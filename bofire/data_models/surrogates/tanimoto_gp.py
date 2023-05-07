from typing import Literal

from pydantic import Field, validator

from bofire.data_models.enum import MolecularEncodingEnum

from bofire.data_models.kernels.api import AnyKernel, TanimotoKernel, ScaleKernel
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.features.molecular import MolecularInput


class TanimotoGPSurrogate(BotorchSurrogate):
    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel())
    )

    @validator("input_preprocessing_specs")
    def validate_categoricals(cls, v, values):
        """Checks that at only one of fingerprints, fragments or fingerprints_fragments features are present."""
        if not all([isinstance(x, MolecularInput) for x in values['inputs'].get()]):
            raise ValueError('Found input features other than MolecularInput. TanimotoGPSurrogate can only be used if with molecular fingerprints, fragments or fingerprints_fragments features.')
        if MolecularEncodingEnum.MOL_DESCRIPTOR in v.values():
            raise ValueError(
                "TanimotoGPSurrogate can only be used with molecular fingerprints, fragments or fingerprints_fragments features."
            )
        return v
