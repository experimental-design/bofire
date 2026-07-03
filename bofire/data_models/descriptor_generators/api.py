from bofire.data_models.descriptor_generators.generators import (
    DescriptorGenerator,
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.unions import tagged_union


AnyDescriptorGenerator = tagged_union(
    Fingerprints,
    Fragments,
    MordredDescriptors,
)

__all__ = [
    "AnyDescriptorGenerator",
    "DescriptorGenerator",
    "Fingerprints",
    "Fragments",
    "MordredDescriptors",
]
