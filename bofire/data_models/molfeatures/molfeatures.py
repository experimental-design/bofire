from abc import abstractmethod
from typing import Annotated, List, Literal, Optional

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.molfeatures import names
from bofire.utils.cheminformatics import (  # smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2fragments_fingerprints,
    smiles2mordred,
)


class MolFeatures(BaseModel):
    """Base class for all molecular features"""

    type: str

    @abstractmethod
    def get_descriptor_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        pass


class Fingerprints(MolFeatures):
    type: Literal["Fingerprints"] = "Fingerprints"
    bond_radius: int = 5
    n_bits: int = 2048

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


class Fragments(MolFeatures):
    type: Literal["Fragments"] = "Fragments"
    fragments: Optional[List[str]] = None

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
                    "Not all provided fragments were not found in the RDKit list",
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


class FingerprintsFragments(Fingerprints, Fragments):
    type: Literal["FingerprintsFragments"] = "FingerprintsFragments"

    def get_descriptor_names(self) -> List[str]:
        fingerprints_list = [f"fingerprint_{i}" for i in range(self.n_bits)]
        fragments_list = (
            self.fragments if self.fragments is not None else names.fragments
        )

        fingerprints_fragment_list = fingerprints_list + fragments_list

        return fingerprints_fragment_list

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        fragments_list = (
            self.fragments if self.fragments is not None else names.fragments
        )

        return pd.DataFrame(
            data=smiles2fragments_fingerprints(
                values.to_list(),
                bond_radius=self.bond_radius,
                n_bits=self.n_bits,
                fragments_list=fragments_list,
            ),
            columns=self.get_descriptor_names(),
            index=values.index,
        )


class MordredDescriptors(MolFeatures):
    type: Literal["MordredDescriptors"] = "MordredDescriptors"
    descriptors: Annotated[List[str], Field(min_length=1)]

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
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("descriptors must be unique")

        if not all(desc in names.mordred for desc in descriptors):
            raise ValueError(
                "Not all provided descriptors were not found in the Mordred list",
            )

        return descriptors

    def get_descriptor_names(self) -> List[str]:
        return self.descriptors

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2mordred(values.to_list(), self.descriptors),
            columns=self.descriptors,
            index=values.index,
        )
