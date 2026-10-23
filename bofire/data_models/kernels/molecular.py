from typing import Literal, Optional

from pydantic import Field, model_validator

from bofire.data_models.descriptor_generators.api import Fingerprints, Fragments
from bofire.data_models.encodings.descriptors import DescriptorEncoding
from bofire.data_models.kernels.kernel import FeatureSpecificKernel


class MolecularKernel(FeatureSpecificKernel):
    """Kernel acting on molecular descriptors."""

    pass


class TanimotoKernel(MolecularKernel):
    r"""Kernel over molecular fingerprints, based on the Tanimoto similarity.

    $$
    k(\mathbf x, \mathbf x') = \frac{\mathbf x^{\top}\mathbf x'}
        {\lVert \mathbf x \rVert^2 + \lVert \mathbf x' \rVert^2
         - \mathbf x^{\top}\mathbf x'}
    $$

    Normalizing the shared bits by the bits present in either input is what makes this
    the standard similarity for the sparse binary vectors a fingerprint produces. The
    kernel has no hyperparameters; wrap it in a `ScaleKernel` to fit its scale.

    It is used in one of two ways. With `fingerprints`, it works on one categorical
    input with structure data, encoded as ordinal codes: the similarities between all
    of its categories are computed once, which makes fitting and optimizing fast.
    Without, it works on fingerprint columns that the surrogate's `DescriptorEncoding`
    produces, computing the similarities as needed.

    Examples:
        >>> TanimotoKernel(
        ...     features=["solvent"],
        ...     fingerprints=DescriptorEncoding(
        ...         columns=[], generators=[Fingerprints(n_bits=2048)]
        ...     ),
        ... )
    """

    type: Literal["TanimotoKernel"] = "TanimotoKernel"
    fingerprints: Optional[DescriptorEncoding] = Field(
        default=None,
        description="Fingerprints of the categories of the one categorical input in "
        "`features`, which must then be ordinal-encoded; only Fingerprints and "
        "Fragments generators, no static columns. If not provided, the kernel "
        "works on the fingerprint columns the surrogate's encoding produces.",
    )

    @model_validator(mode="after")
    def validate_fingerprints(self):
        """Fingerprints describe the categories of exactly one input, as bit or count
        vectors."""
        if self.fingerprints is None:
            return self
        if self.features is None or len(self.features) != 1:
            raise ValueError(
                "A Tanimoto kernel with fingerprints needs exactly one feature."
            )
        if (
            self.fingerprints.columns
            or not self.fingerprints.generators
            or not all(
                isinstance(generator, (Fingerprints, Fragments))
                for generator in self.fingerprints.generators
            )
        ):
            raise ValueError(
                "The fingerprints of a Tanimoto kernel must come from Fingerprints "
                "or Fragments generators only, without static columns."
            )
        return self
