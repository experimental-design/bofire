from typing import Union

from bofire.data_models.molfeatures.types import Fingerprints, Fragments, FingerprintsFragments, MordredDescriptors # BagOfCharacters
from bofire.data_models.molfeatures.molfeatures import MolFeatures

AbstractMolFeatures = MolFeatures

AnyMolFeatures = Union[
    Fingerprints,
    Fragments,
    FingerprintsFragments,
    # BagOfCharacters,
    MordredDescriptors,
]
