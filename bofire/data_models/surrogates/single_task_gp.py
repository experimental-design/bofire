from typing import Literal

from pydantic import Field, validator

from bofire.data_models.kernels.api import AnyKernel, MaternKernel, ScaleKernel
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

    @validator("scaler")
    def validate_scaler(cls, v, values):
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
        for l in dict_lists:
            if "TanimotoKernel" in l:
                if v != ScalerEnum.IDENTITY:
                    raise ValueError(
                        "Must use ScalerEnum.IDENTITY when using TanimotoKernel"
                    )
        return v
