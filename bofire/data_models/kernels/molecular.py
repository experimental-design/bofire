from typing import Literal

from bofire.data_models.kernels.kernel import ARDKernel, FeatureSpecificKernel


class MolecularKernel(FeatureSpecificKernel):
    """Kernel acting on molecular descriptors."""

    pass


class TanimotoKernel(ARDKernel, MolecularKernel):
    """Kernel measuring molecular similarity as the overlap of two fingerprints.

    Compares which descriptor bits two molecules share relative to how many they have
    between them, which is the standard similarity measure for fingerprints and suits
    the sparse binary vectors they produce.
    """

    type: Literal["TanimotoKernel"] = "TanimotoKernel"
