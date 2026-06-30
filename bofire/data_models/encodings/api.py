from bofire.data_models.encodings.descriptors import DescriptorEncoding
from bofire.data_models.encodings.molecular import MolecularEncoding
from bofire.data_models.unions import tagged_union


_ENCODING_TYPES = [DescriptorEncoding, MolecularEncoding]

AnyCategoricalEncoding = tagged_union(*_ENCODING_TYPES)
