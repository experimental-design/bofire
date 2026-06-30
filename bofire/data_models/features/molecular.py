import warnings
from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator, validate_call

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.utils.cheminformatics import smiles2mol


class ContinuousMolecularInput(ContinuousInput):
    type: Literal["ContinuousMolecularInput"] = "ContinuousMolecularInput"
    order_id: ClassVar[int] = 4
    molecule: str

    def _description_prefix(self) -> str:
        return (
            f"Continuous molecular (SMILES: {self.molecule}), "
            f"bounds [{self.bounds[0]}, {self.bounds[1]}]"
        )

    @field_validator("molecule")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Validates that molecule is a valid smiles. Note that this check can only
        be executed when rdkit is available.

        Args:
            v (str): smiles
        """
        # check on rdkit availability:
        try:
            smiles2mol(v)
        except NameError:
            warnings.warn("rdkit not installed, molecule cannot be validated.")
            return v
        smiles2mol(v)
        return v


class CategoricalMolecularInput(CategoricalInput):
    """Deprecated. Use :class:`CategoricalInput` with a ``smiles`` descriptor column
    and a :class:`MolecularEncoding` on the surrogate instead.

    Kept as a thin deserialization shim: the SMILES categories are mirrored into a
    reserved ``smiles`` descriptor column so molecular encoders can consume them.
    """

    type: Literal["CategoricalMolecularInput"] = "CategoricalMolecularInput"
    # order_id: ClassVar[int] = 7
    order_id: ClassVar[int] = 5

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_smiles(cls, data):
        if not isinstance(data, dict):
            return data
        warnings.warn(
            "`CategoricalMolecularInput` is deprecated, use `CategoricalInput` with "
            "a `smiles` descriptor column and a `MolecularEncoding` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        categories = data.get("categories")
        descriptors = dict(data.get("descriptors") or {})
        if categories is not None and "smiles" not in descriptors:
            descriptors["smiles"] = list(categories)
            data["descriptors"] = descriptors
        return data

    def _description_prefix(self) -> str:
        return (
            f"Categorical molecular (SMILES), allowed: {self.get_allowed_categories()}"
        )

    @field_validator("categories")
    @classmethod
    def validate_smiles(cls, categories: Sequence[str]):
        """Validates that categories are valid smiles. Note that this check can only
        be executed when rdkit is available.

        Args:
            categories (List[str]): List of smiles

        Raises:
            ValueError: when string is not a smiles

        Returns:
            List[str]: List of the smiles

        """
        # check on rdkit availability:
        try:
            smiles2mol(categories[0])
        except NameError:
            warnings.warn("rdkit not installed, categories cannot be validated.")
            return categories

        for cat in categories:
            smiles2mol(cat)
        return categories

    @validate_call
    def select_mordred_descriptors(
        self,
        transform_type: MordredDescriptors,
        cutoff: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95,
    ) -> None:
        """
        Filter Mordred descriptors by removing highly correlated ones.

        This function removes descriptors with zero variance and then iteratively
        filters out descriptors that are highly correlated with already selected ones.
        Uses a greedy algorithm that iteratively selects descriptors and removes those
        that are highly correlated with the selected ones.

        Args:
            transform_type: MordredDescriptors object containing the initial list of descriptors
            cutoff: Absolute correlation threshold above which descriptors are considered redundant. Range: [0.0, 1.0]. Default: 0.95

        Raises:
            ValueError: If no descriptors with non-zero variance are found

        """  # noqa: W293
        # Get unique SMILES to avoid redundant calculations
        unique_smiles = pd.Series(self.get_allowed_categories())

        # Get descriptor values for all SMILES
        descriptor_values = transform_type.get_descriptor_values(unique_smiles)

        # Remove columns with zero variance (non-informative descriptors)
        variances = descriptor_values.var()
        non_zero_var_descriptors = variances[variances > 0].index.tolist()

        if len(non_zero_var_descriptors) == 0:
            raise ValueError(
                f"No descriptors with non-zero variance found for feature '{self.key}'. "
                "Cannot perform correlation-based filtering."
            )

        descriptor_values = descriptor_values[non_zero_var_descriptors]

        # Handle edge case: only one descriptor left
        if descriptor_values.shape[1] == 1:
            warnings.warn(
                f"Only one descriptor with non-zero variance found for feature '{self.key}'. "
                "No correlation filtering needed."
            )
            transform_type.descriptors = non_zero_var_descriptors
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
                if correlation_matrix.iloc[current_idx, idx] > cutoff:
                    to_remove.add(idx)

            remaining_descriptors -= to_remove

        # Update the transform_type.descriptors with the filtered list
        transform_type.descriptors = selected_descriptors
