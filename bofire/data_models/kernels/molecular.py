from typing import Literal

from bofire.data_models.kernels.kernel import FeatureSpecificKernel


class MolecularKernel(FeatureSpecificKernel):
    pass


class TanimotoKernel(MolecularKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    ard: bool = True
