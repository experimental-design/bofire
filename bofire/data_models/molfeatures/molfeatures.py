import warnings
from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.molfeatures import names
from bofire.utils.cheminformatics import (  # smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mordred,
)


class MolFeatures(BaseModel):
    """Base class for all molecular features.

    A ``MolFeatures`` is a pure, stateless transform: given a series of structure
    identifiers (SMILES) it returns a dataframe of numeric descriptor columns.
    Correlation-based decorrelation is *not* done here — it lives on the consuming
    descriptor encoding / engineered feature (``DescriptorSpec``), applied across the
    whole assembled descriptor block.
    """

    type: Any
    # the kind of structure identifier this generator consumes; the descriptor spec
    # validates its structure column carries this kind.
    reads: Literal["smiles"] = "smiles"

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_filter_fields(cls, data):
        """Correlation filtering moved to the descriptor encoding/feature.

        Old serialized generators may still carry ``filter_descriptors`` /
        ``correlation_cutoff``; drop them (with a warning) so those dumps still load.
        """
        if isinstance(data, dict) and (
            "filter_descriptors" in data or "correlation_cutoff" in data
        ):
            warnings.warn(
                "`filter_descriptors` / `correlation_cutoff` moved off MolFeatures to "
                "the descriptor encoding/feature (`filter_descriptors` on "
                "`DescriptorEncoding` / `WeightedSumFeature`); the values on the "
                "generator are ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
            data = {
                k: v
                for k, v in data.items()
                if k not in ("filter_descriptors", "correlation_cutoff")
            }
        return data

    @abstractmethod
    def get_descriptor_names(self) -> List[str]:
        """The descriptor column names this generator produces."""

    @abstractmethod
    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        """Descriptor values for a series of structures (columns = descriptor names)."""


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
    fragments: Optional[Annotated[List[str], Field(min_length=1)]] = None

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


class MordredDescriptors(MolFeatures):
    type: Literal["MordredDescriptors"] = "MordredDescriptors"
    descriptors: Optional[Annotated[List[str], Field(min_length=1)]] = None
    ignore_3D: bool = False

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
                    "Not all provided descriptors were not found in the Mordred list",
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
