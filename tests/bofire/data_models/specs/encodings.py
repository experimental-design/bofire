import bofire.data_models.descriptor_generators.api as descriptor_generators
import bofire.data_models.encodings.api as encodings
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    encodings.OneHotEncoding,
    lambda: {"drop_first": False},
)

specs.add_valid(
    encodings.OneHotEncoding,
    lambda: {"drop_first": True},
)

specs.add_valid(
    encodings.OrdinalEncoding,
    lambda: {},
)


def _descriptor_spec(**kwargs):
    """A DescriptorEncoding spec with all mixin defaults spelled out."""
    return {
        "columns": None,
        "generators": [],
        "filter_descriptors": False,
        "correlation_cutoff": 0.95,
        **kwargs,
    }


# static descriptor columns (all / subset)
specs.add_valid(encodings.DescriptorEncoding, lambda: _descriptor_spec())
specs.add_valid(
    encodings.DescriptorEncoding, lambda: _descriptor_spec(columns=["d1", "d2"])
)

# molecular generator
specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=[],
        generators=[
            descriptor_generators.Fingerprints(n_bits=32, bond_radius=3).model_dump()
        ],
    ),
)

# composite: static + molecular on one feature
specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=["logP"],
        generators=[
            descriptor_generators.Fingerprints(n_bits=32, bond_radius=3).model_dump()
        ],
        filter_descriptors=True,
    ),
)

# two generators on the same structure column
specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=[],
        generators=[
            descriptor_generators.Fingerprints(n_bits=32, bond_radius=3).model_dump(),
            descriptor_generators.Fragments().model_dump(),
        ],
    ),
)


# A spec must be coherent on its own: the names it declares -- the columns it lists plus
# the ones its generators emit -- have to be unique. A collision that only shows up once
# `columns=None` is resolved against a feature is the gate's business, not this one's.
specs.add_invalid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=[],
        generators=[
            descriptor_generators.Fingerprints(n_bits=8).model_dump(),
            descriptor_generators.Fingerprints(n_bits=8).model_dump(),
        ],
    ),
    error=ValueError,
    message="descriptor names must be unique",
)

specs.add_invalid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(columns=["logP", "logP"]),
    error=ValueError,
    message="descriptor names must be unique",
)

specs.add_invalid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=["fingerprint_0"],
        generators=[descriptor_generators.Fingerprints(n_bits=8).model_dump()],
    ),
    error=ValueError,
    message="descriptor names must be unique",
)
