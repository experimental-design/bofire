from typing import Union

from bofire.models.gps import kernels

AnyKernel = Union[
    kernels.HammondDistanceKernel,
    kernels.RBFKernel,
    kernels.MaternKernel,
    kernels.LinearKernel,
    kernels.ScaleKernel,
    kernels.MultiplicativeKernel,
    kernels.AdditiveKernel,
]
