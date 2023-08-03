from typing import Literal, Optional, Sequence, Union

from bofire.data_models.kernels.categorical import HammondDistanceKernel
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
            HammondDistanceKernel,
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
            HammondDistanceKernel,
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
        HammondDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        TanimotoKernel,
        "ScaleKernel",
    ]
    outputscale_prior: Optional[AnyPrior] = None


AdditiveKernel.update_forward_refs()
MultiplicativeKernel.update_forward_refs()
