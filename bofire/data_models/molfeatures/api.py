from bofire.data_models.molfeatures.molfeatures import (
    Fingerprints,
    Fragments,
    MolFeatures,
    MordredDescriptors,
)
from bofire.data_models.unions import tagged_union


AnyMolFeatures = tagged_union(
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
