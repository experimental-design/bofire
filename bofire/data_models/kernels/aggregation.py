from typing import Literal, Optional, Sequence, Union

from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.data_models.kernels.continuous import LinearKernel, MaternKernel, RBFKernel
from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.priors.api import AnyPrior


class AdditiveKernel(Kernel):
    type: Literal["AdditiveKernel"] = "AdditiveKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            TanimotoKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]
    type: Literal["AdditiveKernel"] = "AdditiveKernel"


class MultiplicativeKernel(Kernel):
    type: Literal["MultiplicativeKernel"] = "MultiplicativeKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammingDistanceKernel,
            AdditiveKernel,
            TanimotoKernel,
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]


class ScaleKernel(Kernel):
    type: Literal["ScaleKernel"] = "ScaleKernel"
    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammingDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        TanimotoKernel,
        "ScaleKernel",
    ]
    outputscale_prior: Optional[AnyPrior] = None


AdditiveKernel.model_rebuild()
MultiplicativeKernel.model_rebuild()
