import random

import bofire.data_models.descriptor_generators.api as descriptor_generators
from bofire.data_models.descriptor_generators import names
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

# None of these need rdkit. A generator needs it to produce descriptor *values*, which
# specs never do -- they construct and serialize. The name lists sampled from below are
# available either way: `names.fragments` is a plain list and `names.mordred` falls back
# to a hardcoded one. Leaving them ungated is what gets all three covered by the
# bare-install CI job.
specs.add_valid(
    descriptor_generators.Fingerprints,
    lambda: {
        "bond_radius": random.randrange(1, 6),
        "n_bits": random.randrange(32, 2048),
    },
)

specs.add_valid(
    descriptor_generators.Fragments,
    lambda: {
        "fragments": random.sample(
            names.fragments,
            k=random.randrange(1, len(names.fragments)),
        ),
    },
)

specs.add_valid(
    descriptor_generators.MordredDescriptors,
    lambda: {
        "descriptors": random.sample(names.mordred, k=random.randrange(1, 10)),
        "ignore_3D": False,
    },
)
