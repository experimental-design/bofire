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

specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: {"columns": None},
)

specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: {"columns": ["d1", "d2"]},
)

specs.add_valid(
    encodings.MolecularEncoding,
    lambda: {
        "structure": "smiles",
        "generator": molfeatures.Fingerprints(
            n_bits=32,
            bond_radius=3,
        ).model_dump(),
    },
)
