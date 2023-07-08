from typing import Union

from bofire.data_models.molfeatures.molfeatures import (  # BagOfCharacters
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
    # BagOfCharacters,
    MordredDescriptors,
]
