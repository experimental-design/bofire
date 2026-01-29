from typing import Union

from bofire.data_models.molfeatures.molfeatures import (
    CompositeMolFeatures,
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MolFeatures,
    MordredDescriptors,
)


AbstractMolFeatures = MolFeatures

AnyMolFeatures = Union[
    CompositeMolFeatures,
    Fingerprints,
    Fragments,
    MordredDescriptors,
]
