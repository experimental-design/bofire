import bofire.data_models.encodings.api as encodings
import bofire.data_models.molfeatures.api as molfeatures
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
        "generators": {},
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
        generators={
            "smiles": [molfeatures.Fingerprints(n_bits=32, bond_radius=3).model_dump()]
        },
    ),
)

# composite: static + molecular on one feature
specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=["logP"],
        generators={
            "smiles": [molfeatures.Fingerprints(n_bits=32, bond_radius=3).model_dump()]
        },
        filter_descriptors=True,
    ),
)

# two generators on the same structure column
specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: _descriptor_spec(
        columns=[],
        generators={
            "smiles": [
                molfeatures.Fingerprints(n_bits=32, bond_radius=3).model_dump(),
                molfeatures.Fragments().model_dump(),
            ]
        },
    ),
)
