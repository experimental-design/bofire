import warnings
from abc import abstractmethod
from typing import Annotated, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field, PrivateAttr, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.molfeatures import names
from bofire.utils.cheminformatics import (  # smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mordred,
)


class MolFeatures(BaseModel):
    """Base class for all molecular features"""

    type: str
    filter_descriptors: bool = True
    correlation_cutoff: float = 0.95
    _descriptors: Optional[Annotated[List[str], Field(min_length=1)]] = PrivateAttr(
        None
    )

    def get_descriptor_names(self) -> List[str]:
        if self._descriptors is not None:
            return self._descriptors
        return self._get_descriptor_names()

    @abstractmethod
    def _get_descriptor_names(self) -> List[str]:
        pass

    def get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        all_descriptor_values = self._get_descriptor_values(values)
        if self._descriptors is not None:
            return all_descriptor_values[self._descriptors].copy()
        return all_descriptor_values

    @abstractmethod
    def _get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        pass

    def remove_correlated_descriptors(self, molecules: List[str]):
        # Get unique SMILES to avoid redundant calculations
        unique_smiles = pd.Series(molecules)

        # Get descriptor values for all SMILES
        descriptor_values = self.get_descriptor_values(unique_smiles)

        # Remove columns with zero variance (non-informative descriptors)
        variances = descriptor_values.var()
        non_zero_var_descriptors = variances[variances > 0].index.tolist()

        if len(non_zero_var_descriptors) == 0:
            raise ValueError(
                "No descriptors with non-zero variance found. "
                "Cannot perform correlation-based filtering."
            )

        descriptor_values = descriptor_values[non_zero_var_descriptors]

        # Handle edge case: only one descriptor left
        if descriptor_values.shape[1] == 1:
            warnings.warn(
                "Only one descriptor with non-zero variance found for feature. "
                "No correlation filtering needed."
            )
            self._descriptors = non_zero_var_descriptors
            return

        # Compute absolute correlation matrix
        correlation_matrix = descriptor_values.corr().abs()

        # Greedy algorithm to select uncorrelated descriptors
        selected_descriptors = []
        remaining_descriptors = set(range(len(descriptor_values.columns)))

        while remaining_descriptors:
            # Select the first remaining descriptor
            current_idx = min(remaining_descriptors)
            selected_descriptors.append(descriptor_values.columns[current_idx])
            remaining_descriptors.remove(current_idx)

            # Find and remove highly correlated descriptors
            to_remove = set()
            for idx in remaining_descriptors:
                if correlation_matrix.iloc[current_idx, idx] > self.correlation_cutoff:
                    to_remove.add(idx)

            remaining_descriptors -= to_remove

        # Update the transform_type.descriptors with the filtered list
        if self.filter_descriptors is True:
            self._descriptors = selected_descriptors
        return selected_descriptors


class Fingerprints(MolFeatures):
    type: Literal["Fingerprints"] = "Fingerprints"
    bond_radius: int = 5
    n_bits: int = 2048

    def _get_descriptor_names(self) -> List[str]:
        return [f"fingerprint_{i}" for i in range(self.n_bits)]

    def _get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2fingerprints(
                values.to_list(),
                bond_radius=self.bond_radius,
                n_bits=self.n_bits,
            ).astype(float),
            columns=self._get_descriptor_names(),
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

    def _get_descriptor_names(self) -> List[str]:
        return self.fragments if self.fragments is not None else names.fragments

    def _get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2fragments(values.to_list(), self._get_descriptor_names()),
            columns=self._get_descriptor_names(),
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

    def _get_descriptor_names(self) -> List[str]:
        return self.descriptors if self.descriptors is not None else names.mordred

    def _get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=smiles2mordred(
                values.to_list(), self._get_descriptor_names(), ignore_3D=self.ignore_3D
            ),
            columns=self._get_descriptor_names(),
            index=values.index,
        )


class CompositeMolFeatures(MolFeatures):
    """Composite of multiple MolFeatures objects.

    Combines descriptor sets from multiple MolFeatures into a single
    feature space. Descriptor names across components must be unique.
    """

    type: Literal["CompositeMolFeatures"] = "CompositeMolFeatures"
    features: Annotated[
        List[Union[Fingerprints, Fragments, MordredDescriptors]], Field(min_length=2)
    ]

    def _get_descriptor_names(self) -> List[str]:
        all_names: List[str] = []
        for feat in self.features:
            all_names.extend(feat._get_descriptor_names())
        duplicate_names = {name for name in all_names if all_names.count(name) > 1}
        if duplicate_names:
            raise ValueError(
                "Duplicate descriptor names found in CompositeMolFeatures: "
                f"{sorted(duplicate_names)}",
            )
        return all_names

    def _get_descriptor_values(self, values: pd.Series) -> pd.DataFrame:
        descriptor_dfs = [feat._get_descriptor_values(values) for feat in self.features]
        combined = pd.concat(descriptor_dfs, axis=1)
        if combined.columns.duplicated().any():
            duplicate_names = combined.columns[combined.columns.duplicated()].tolist()
            raise ValueError(
                "Duplicate descriptor names found in CompositeMolFeatures: "
                f"{sorted(set(duplicate_names))}",
            )
        return combined


def FingerprintsFragments(
    fragments: Optional[List[str]] = None,
    bond_radius: int = 5,
    n_bits: int = 2048,
    correlation_cutoff: float = 0.95,
    filter_descriptors: bool = True,
) -> CompositeMolFeatures:
    """Factory function to create a FingerprintsFragments MolFeatures object.

    Args:
        fragments (Optional[List[str]], optional): List of fragment names. Defaults to None.
        bond_radius (int, optional): Bond radius for fingerprints. Defaults to 5.
        n_bits (int, optional): Number of bits for fingerprints. Defaults to 2048.

    Returns:
        MolFeatures: An instance of FingerprintsFragments.
    """
    return CompositeMolFeatures(
        correlation_cutoff=correlation_cutoff,
        features=[
            Fingerprints(bond_radius=bond_radius, n_bits=n_bits),
            Fragments(fragments=fragments),
        ],
        filter_descriptors=filter_descriptors,
    )
