import importlib
import random

import bofire.data_models.descriptor_generators.api as descriptor_generators
from bofire.data_models.descriptor_generators import names
from tests.bofire.data_models.specs.specs import Specs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None
MORDRED_AVAILABLE = importlib.util.find_spec("mordred") is not None

specs = Specs([])

specs.add_valid(
    descriptor_generators.Fingerprints,
    lambda: {
        "bond_radius": random.randrange(1, 6),
        "n_bits": random.randrange(32, 2048),
    },
)

if RDKIT_AVAILABLE:
    specs.add_valid(
        descriptor_generators.Fragments,
        lambda: {
            "fragments": random.sample(
                names.fragments,
                k=random.randrange(1, len(names.fragments)),
            ),
        },
    )

    if MORDRED_AVAILABLE:
        specs.add_valid(
            descriptor_generators.MordredDescriptors,
            lambda: {
                "descriptors": random.sample(names.mordred, k=random.randrange(1, 10)),
                "ignore_3D": False,
            },
        )
