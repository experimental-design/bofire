from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional

import pandas as pd
from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.descriptor_generators import names
from bofire.utils.cheminformatics import (  # smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mordred,
)


class DescriptorGenerator(BaseModel):
    """Computes numeric descriptor columns from molecular structures (SMILES).

    The same structure always gives the same descriptors.
    """

    type: Any

    @abstractmethod
    def get_descriptor_names(self) -> List[str]:
        """The descriptor column names this generator produces."""

    @abstractmethod
    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        """Descriptor values for a series of structures (columns = descriptor names)."""


class Fingerprints(DescriptorGenerator):
    """Morgan (circular) fingerprints: one bit per hashed substructure, set if the
    substructure occurs in the molecule."""

    type: Literal["Fingerprints"] = "Fingerprints"
    bond_radius: PositiveInt = Field(
        default=5,
        description="Radius, in bonds, of the substructures around each atom.",
    )
    n_bits: PositiveInt = Field(
        default=2048,
        description="Length of the fingerprint. Fewer bits mean more distinct "
        "substructures share a bit.",
    )

    def get_descriptor_names(self) -> List[str]:
        return [f"fingerprint_{i}" for i in range(self.n_bits)]

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2fingerprints(
                values.to_list(),
                bond_radius=self.bond_radius,
                n_bits=self.n_bits,
            ).astype(float),
            columns=self.get_descriptor_names(),
            index=values.index,
        )


class Fragments(DescriptorGenerator):
    """Counts of functional groups and other substructures, as RDKit's fragment
    descriptors (`fr_...`) count them."""

    type: Literal["Fragments"] = "Fragments"
    fragments: Optional[Annotated[List[str], Field(min_length=1)]] = Field(
        default=None,
        description="Names of the RDKit fragment descriptors to compute, e.g. "
        "'fr_Al_OH'. If not provided, all 85 of them.",
    )

    @field_validator("fragments")
    @classmethod
    def validate_fragments(cls, fragments):
        """Validates that fragments have unique names

        Args:
            categories (List[str]): List of fragment names

        Raises:
            ValueError: when fragments have non-unique names

        Returns:
            List[str]: List of the fragments

        """
        if fragments is not None:
            if len(fragments) != len(set(fragments)):
                raise ValueError("Fragments must be unique")

            if not all(user_fragment in names.fragments for user_fragment in fragments):
                raise ValueError(
                    "Unknown fragments, not in RDKit's list: "
                    f"{sorted(set(fragments) - set(names.fragments))}",
                )

        return fragments

    def get_descriptor_names(self) -> List[str]:
        return self.fragments if self.fragments is not None else names.fragments

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2fragments(values.to_list(), self.get_descriptor_names()),
            columns=self.get_descriptor_names(),
            index=values.index,
        )


class MordredDescriptors(DescriptorGenerator):
    """Molecular descriptors computed by Mordred, e.g. molecular weight, logP or
    topological indices; real-valued, unlike fingerprints and fragment counts."""

    type: Literal["MordredDescriptors"] = "MordredDescriptors"
    descriptors: Optional[Annotated[List[str], Field(min_length=1)]] = Field(
        default=None,
        description="Names of the Mordred descriptors to compute. If not provided, "
        "all 1826 of them.",
    )
    ignore_3D: bool = Field(
        default=False,
        description="Whether to skip descriptors that need a 3D conformation.",
    )

    @field_validator("descriptors")
    @classmethod
    def validate_descriptors(cls, descriptors):
        """Validates that descriptors have unique names

        Args:
            descriptors (List[str]): List of descriptor names

        Raises:
            ValueError: when descriptors have non-unique names

        Returns:
            List[str]: List of the descriptors

        """
        if descriptors is not None:
            if len(descriptors) != len(set(descriptors)):
                raise ValueError("descriptors must be unique")

            if not all(desc in names.mordred for desc in descriptors):
                raise ValueError(
                    "Unknown descriptors, not in Mordred's list: "
                    f"{sorted(set(descriptors) - set(names.mordred))}",
                )
        return descriptors

    def get_descriptor_names(self) -> List[str]:
        return self.descriptors if self.descriptors is not None else names.mordred

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2mordred(
                values.to_list(), self.get_descriptor_names(), ignore_3D=self.ignore_3D
            ),
            columns=self.get_descriptor_names(),
            index=values.index,
        )
