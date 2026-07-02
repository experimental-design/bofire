from bofire.data_models.descriptors.composite import CompositeSource
from bofire.data_models.descriptors.generated import GeneratedSource
from bofire.data_models.descriptors.source import DescriptorSource
from bofire.data_models.descriptors.static import StaticSource
from bofire.data_models.unions import tagged_union


_SOURCE_TYPES = [StaticSource, GeneratedSource, CompositeSource]

AnyDescriptorSource = tagged_union(*_SOURCE_TYPES)

__all__ = [
    "AnyDescriptorSource",
    "CompositeSource",
    "DescriptorSource",
    "GeneratedSource",
    "StaticSource",
]
