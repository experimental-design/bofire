from typing import Union

from bofire.data_models.molfeatures.molfeatures import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MolFeatures,
    MordredDescriptors,
)


AbstractMolFeatures = MolFeatures

AnyMolFeatures = Union[
    Fingerprints,
    Fragments,
    FingerprintsFragments,
    MordredDescriptors,
]
