from typing import Literal

from bofire.data_models.kernels.kernel import ConcreteKernel


class MolecularKernel(ConcreteKernel):
    pass


class TanimotoKernel(MolecularKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    ard: bool = True
