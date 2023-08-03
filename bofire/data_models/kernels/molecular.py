from typing import Literal

from bofire.data_models.kernels.kernel import Kernel


class MolecularKernel(Kernel):
    pass


class TanimotoKernel(MolecularKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    ard: bool = True
