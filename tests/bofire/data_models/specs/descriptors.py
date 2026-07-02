import bofire.data_models.descriptors.api as descriptors
import bofire.data_models.molfeatures.api as molfeatures
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    descriptors.StaticSource,
    lambda: {"columns": None},
)

specs.add_valid(
    descriptors.StaticSource,
    lambda: {"columns": ["d1", "d2"]},
)

specs.add_valid(
    descriptors.GeneratedSource,
    lambda: {
        "structure": "smiles",
        "generator": molfeatures.Fingerprints(
            n_bits=32,
            bond_radius=3,
        ).model_dump(),
    },
)

specs.add_valid(
    descriptors.CompositeSource,
    lambda: {
        "sources": [
            descriptors.StaticSource(columns=["logP"]).model_dump(),
            descriptors.GeneratedSource(
                generator=molfeatures.Fingerprints(n_bits=32, bond_radius=3),
            ).model_dump(),
        ],
    },
)
