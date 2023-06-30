from typing import Union

from bofire.data_models.molfeatures.molfeatures import MolFeatures
from bofire.data_models.molfeatures.types import (  # BagOfCharacters
    Fingerprints,
    FingerprintsFragments,
    Fragments,
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
