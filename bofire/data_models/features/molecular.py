import warnings
from collections.abc import Sequence
from typing import Annotated, ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, validate_call

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.feature import get_encoded_name
from bofire.data_models.molfeatures.api import (
    AnyMolFeatures,
    CompositeMolFeatures,
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
from bofire.utils.cheminformatics import smiles2mol


class ContinuousMolecularInput(ContinuousInput):
    type: Literal["ContinuousMolecularInput"] = "ContinuousMolecularInput"
    order_id: ClassVar[int] = 4
    molecule: str

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
    type: Literal["CategoricalMolecularInput"] = "CategoricalMolecularInput"
    # order_id: ClassVar[int] = 7
    order_id: ClassVar[int] = 5

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

    @staticmethod
    def valid_transform_types() -> List[Union[AnyMolFeatures, CategoricalEncodingEnum]]:
        return (
            CategoricalInput.valid_transform_types()
            + [  # ty: ignore[invalid-return-type]
                Fingerprints,
                CompositeMolFeatures,
                Fragments,
                MordredDescriptors,
            ]
        )

    def get_bounds(
        self,
        transform_type: Union[CategoricalEncodingEnum, AnyMolFeatures],
        values: Optional[pd.Series] = None,
        reference_value: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:
        if isinstance(transform_type, CategoricalEncodingEnum):
            # we are just using the standard categorical transformations
            return super().get_bounds(
                transform_type=transform_type,
                values=values,
                reference_value=reference_value,
            )
        # in case that values is None, we return the optimization bounds
        # else we return the complete bounds
        data = self.to_descriptor_encoding(
            transform_type=transform_type,
            values=(
                pd.Series(self.get_allowed_categories())
                if values is None
                else pd.Series(self.categories)
            ),
        )
        lower = data.min(axis=0).values.tolist()
        upper = data.max(axis=0).values.tolist()
        return lower, upper

    def to_descriptor_encoding(
        self,
        transform_type: AnyMolFeatures,
        values: pd.Series,
    ) -> pd.DataFrame:
        """Converts values to descriptor encoding.

        Args:
            values (pd.Series): Values to transform.

        Returns:
            pd.DataFrame: Descriptor encoded dataframe.

        """
        descriptor_values = transform_type.get_descriptor_values(values)

        descriptor_values.columns = [
            get_encoded_name(self.key, d) for d in transform_type.get_descriptor_names()
        ]
        descriptor_values.index = values.index

        return descriptor_values

    def from_descriptor_encoding(
        self,
        transform_type: AnyMolFeatures,
        values: pd.DataFrame,
    ) -> pd.Series:
        """Converts values back from descriptor encoding.

        Args:
            values (pd.DataFrame): Descriptor encoded dataframe.

        Raises:
            ValueError: If descriptor columns not found in the dataframe.

        Returns:
            pd.Series: Series with categorical values.

        """
        # This method is modified based on the categorical descriptor feature
        # TODO: move it to more central place
        cat_cols = [
            get_encoded_name(self.key, d) for d in transform_type.get_descriptor_names()
        ]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}.",
            )
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - self.to_descriptor_encoding(
                            transform_type=transform_type,
                            values=pd.Series(self.get_allowed_categories()),
                        ).to_numpy()
                    )
                    ** 2,
                    axis=2,
                ),
            ),
            columns=self.get_allowed_categories(),
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s

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
