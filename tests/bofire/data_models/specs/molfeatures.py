import importlib
import random

import bofire.data_models.molfeatures.api as molfeatures
from bofire.data_models.molfeatures import names
from tests.bofire.data_models.specs.specs import Specs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None
MORDRED_AVAILABLE = importlib.util.find_spec("mordred") is not None

specs = Specs([])

specs.add_valid(
    molfeatures.Fingerprints,
    lambda: {
        "bond_radius": random.randrange(1, 6),
        "n_bits": random.randrange(32, 2048),
    },
)

if RDKIT_AVAILABLE:
    specs.add_valid(
        molfeatures.Fragments,
        lambda: {
            "fragments": random.sample(
                names.fragments,
                k=random.randrange(1, len(names.fragments)),
            ),
        },
    )
    specs.add_valid(
        molfeatures.FingerprintsFragments,
        lambda: {
            "bond_radius": random.randrange(1, 6),
            "n_bits": random.randrange(32, 2048),
            "fragments": random.sample(
                names.fragments,
                k=random.randrange(1, len(names.fragments)),
            ),
        },
    )

    if MORDRED_AVAILABLE:
        specs.add_valid(
            molfeatures.MordredDescriptors,
            lambda: {
                "descriptors": random.sample(names.mordred, k=random.randrange(1, 10)),
            },
        )
