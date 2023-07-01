from typing import Literal

from pydantic import Field, root_validator

from bofire.data_models.features.api import MolecularInput
from bofire.data_models.kernels.api import AnyKernel, MaternKernel, ScaleKernel
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
)
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class SingleTaskGPSurrogate(BotorchSurrogate):
    type: Literal["SingleTaskGPSurrogate"] = "SingleTaskGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=MaternKernel(
                ard=True,
                nu=2.5,
                lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR(),
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @root_validator
    def validate_scaler_and_kernel(cls, values):
        # Identify if TanimotoKernel is used at any point in the kernel
        def dict_generator(dic, pre=None):
            pre = pre[:] if pre else []
            if isinstance(dic, dict):
                for key, value in dic.items():
                    if isinstance(value, dict):
                        yield from dict_generator(value, pre + [key])
                    elif isinstance(value, (list, tuple)):
                        for vv in value:
                            yield from dict_generator(vv, pre + [key])
                    else:
                        yield pre + [key, value]
            else:
                yield pre + [dic]

        dict_lists = dict_generator(values["kernel"].dict())
        tanimoto_bool_list = []
        for dict_item in dict_lists:
            if "TanimotoKernel" in dict_item:
                tanimoto_bool_list.append(True)
            else:
                tanimoto_bool_list.append(False)

        molfeatures_list = [
            i.molfeatures for i in values["inputs"].get(MolecularInput).features
        ]

        if any(tanimoto_bool_list):
            if values["scaler"] != ScalerEnum.IDENTITY:
                raise ValueError("Use ScalerEnum.IDENTITY when using TanimotoKernel")

        if any(
            (
                isinstance(molfeat, Fingerprints)
                or isinstance(molfeat, FingerprintsFragments)
                or isinstance(molfeat, Fragments)
            )
            for molfeat in molfeatures_list
        ):
            if not any(tanimoto_bool_list):
                raise ValueError(
                    "Use Tanimoto kernel when using fingerprints, fragments, or fingerprints_fragments molecular features"
                )

        return values
