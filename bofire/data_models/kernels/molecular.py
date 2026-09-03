from typing import Literal

from bofire.data_models.kernels.kernel import ARDKernel, FeatureSpecificKernel


class MolecularKernel(FeatureSpecificKernel):
    """Kernel acting on molecular descriptors."""

    pass


class TanimotoKernel(ARDKernel, MolecularKernel):
    r"""Kernel over molecular fingerprints, based on the Tanimoto similarity.

    $$
    k(\mathbf x, \mathbf x') = \frac{\mathbf x^{\top}\mathbf x'}
        {\lVert \mathbf x \rVert^2 + \lVert \mathbf x' \rVert^2
         - \mathbf x^{\top}\mathbf x'}
    $$

    Normalizing the shared bits by the bits present in either input is what makes this
    the standard similarity for the sparse binary vectors a fingerprint produces.
    """

    type: Literal["TanimotoKernel"] = "TanimotoKernel"
