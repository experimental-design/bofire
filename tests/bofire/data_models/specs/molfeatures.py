import importlib
import random
import warnings

import bofire.data_models.molfeatures.api as molfeatures
from tests.bofire.data_models.specs.specs import Specs

try:
    from rdkit.Chem import Descriptors

    fragments_list = [fragment[0] for fragment in Descriptors.descList[124:]]
except ImportError:
    warnings.warn(
        "rdkit not installed, BoFire's cheminformatics utilities cannot be used."
    )

try:
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=False)
    mordred_descriptors = [str(d) for d in calc.descriptors]
except ImportError:
    warnings.warn(
        "mordred not installed. Mordred molecular descriptors cannot be used."
    )

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
                fragments_list, k=random.randrange(1, len(fragments_list))
            )
        },
    )
    specs.add_valid(
        molfeatures.FingerprintsFragments,
        lambda: {
            "bond_radius": random.randrange(1, 6),
            "n_bits": random.randrange(32, 2048),
            "fragments": random.sample(
                fragments_list, k=random.randrange(1, len(fragments_list))
            ),
        },
    )

    if MORDRED_AVAILABLE:
        specs.add_valid(
            molfeatures.MordredDescriptors,
            lambda: {
                "descriptors": random.sample(
                    mordred_descriptors, k=random.randrange(1, 10)
                )
            },
        )
