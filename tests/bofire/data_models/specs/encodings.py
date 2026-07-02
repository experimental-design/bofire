import bofire.data_models.descriptors.api as descriptors
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
    lambda: {"source": descriptors.StaticSource(columns=None).model_dump()},
)

specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: {"source": descriptors.StaticSource(columns=["d1", "d2"]).model_dump()},
)

specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: {
        "source": descriptors.GeneratedSource(
            structure="smiles",
            generator=molfeatures.Fingerprints(n_bits=32, bond_radius=3),
        ).model_dump(),
    },
)

specs.add_valid(
    encodings.DescriptorEncoding,
    lambda: {
        "source": descriptors.CompositeSource(
            sources=[
                descriptors.StaticSource(columns=["logP"]),
                descriptors.GeneratedSource(
                    generator=molfeatures.Fingerprints(n_bits=32, bond_radius=3),
                ),
            ],
        ).model_dump(),
    },
)
