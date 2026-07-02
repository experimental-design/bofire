from bofire.data_models.encodings.descriptors import DescriptorEncoding
from bofire.data_models.encodings.encoding import CategoricalEncoding
from bofire.data_models.encodings.onehot import OneHotEncoding
from bofire.data_models.encodings.ordinal import OrdinalEncoding
from bofire.data_models.unions import tagged_union


_ENCODING_TYPES = [
    OneHotEncoding,
    OrdinalEncoding,
    DescriptorEncoding,
]

AnyCategoricalEncoding = tagged_union(*_ENCODING_TYPES)

__all__ = [
    "AnyCategoricalEncoding",
    "CategoricalEncoding",
    "DescriptorEncoding",
    "OneHotEncoding",
    "OrdinalEncoding",
]
