from typing import Literal

from bofire.data_models.kernels.kernel import ARDKernel, FeatureSpecificKernel


class MolecularKernel(FeatureSpecificKernel):
    """Kernel acting on molecular descriptors.

    Compares molecules by the structural features they share, so it needs a feature
    whose descriptor encoding generates those features -- fingerprints or fragments.
    Handcrafted numeric descriptor columns do not carry that structure.
    """

    @classmethod
    def can_consume(cls, feat, encoding=None) -> bool:
        from bofire.data_models.descriptor_generators.api import Fingerprints, Fragments
        from bofire.data_models.encodings.api import DescriptorEncoding
        from bofire.data_models.features.api import CategoricalInput

        if not isinstance(feat, CategoricalInput) or not isinstance(
            encoding, DescriptorEncoding
        ):
            return False
        return any(
            isinstance(g, (Fingerprints, Fragments)) for g in encoding.generators
        )


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
