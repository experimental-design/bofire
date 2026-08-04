from bofire.data_models.molfeatures.molfeatures import (
    CompositeMolFeatures,
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MolFeatures,
    MordredDescriptors,
)
from bofire.data_models.unions import tagged_union


AnyMolFeatures = tagged_union(
    CompositeMolFeatures,
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
