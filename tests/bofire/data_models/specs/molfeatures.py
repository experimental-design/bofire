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
        "correlation_cutoff": 0.9,
    },
)

if RDKIT_AVAILABLE:
    specs.add_valid(
        molfeatures.Fragments,
        lambda: {
            "correlation_cutoff": 0.9,
            "fragments": random.sample(
                names.fragments,
                k=random.randrange(1, len(names.fragments)),
            ),
        },
    )
    specs.add_valid(
        molfeatures.CompositeMolFeatures,
        lambda: {
            "correlation_cutoff": 0.9,
            "features": [
                molfeatures.Fingerprints(
                    bond_radius=random.randrange(1, 6),
                    n_bits=random.randrange(32, 2048),
                ).model_dump(),
                molfeatures.Fragments(
                    fragments=random.sample(
                        names.fragments,
                        k=random.randrange(1, len(names.fragments)),
                    ),
                ).model_dump(),
            ],
        },
    )

    if MORDRED_AVAILABLE:
        specs.add_valid(
            molfeatures.MordredDescriptors,
            lambda: {
                "correlation_cutoff": 0.9,
                "descriptors": random.sample(names.mordred, k=random.randrange(1, 10)),
                "ignore_3D": False,
            },
        )
