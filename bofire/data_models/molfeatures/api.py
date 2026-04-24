from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.molfeatures.molfeatures import (
    CompositeMolFeatures,
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MolFeatures,
    MordredDescriptors,
)


AbstractMolFeatures = MolFeatures

AnyMolFeatures = Annotated[
    Union[
        CompositeMolFeatures,
        Fingerprints,
        Fragments,
        MordredDescriptors,
    ],
    Field(discriminator="type"),
]
