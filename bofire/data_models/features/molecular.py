import warnings
from collections.abc import Sequence
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import field_validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.feature import Input, get_encoded_name
from bofire.data_models.molfeatures.api import (
    AnyMolFeatures,
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.utils.cheminformatics import smiles2mol


class MolecularInput(Input):
    type: Literal["MolecularInput"] = "MolecularInput"  # type: ignore
    # order_id: ClassVar[int] = 6
    order_id: ClassVar[int] = 4

    @staticmethod
    def valid_transform_types() -> List[AnyMolFeatures]:  # type: ignore
        return [Fingerprints, FingerprintsFragments, Fragments, MordredDescriptors]  # type: ignore

    def validate_experimental(
        self,
        values: pd.Series,
        strict: bool = False,
    ) -> pd.Series:
        values = values.map(str)
        for smi in values:
            smiles2mol(smi)

        return values

    def is_fulfilled(self, values: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "`is_fulfilled` is not implemented for `MolecularInput`. "
            "Please use `CategoricalMolecularInput` instead of `MolecularInput`.",
        )

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        values = values.map(str)
        for smi in values:
            smiles2mol(smi)
        return values

    def is_fixed(self) -> bool:
        return False

    def fixed_value(self, transform_type: Optional[AnyMolFeatures] = None) -> None:  # type: ignore
        return None

    def sample(self, n: int, seed: Optional[int] = None) -> pd.Series:
        raise ValueError("Sampling not supported for `MolecularInput`")

    def get_bounds(  # type: ignore
        self,
        transform_type: AnyMolFeatures,
        values: pd.Series,
        reference_value: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:
        """Calculates the lower and upper bounds for the feature based on the given transform type and values.

        Args:
            transform_type (AnyMolFeatures): The type of transformation to apply to the data.
            values (pd.Series): The actual data over which the lower and upper bounds are calculated.
            reference_value (Optional[str], optional): The reference value for the transformation. Not used here.
                Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing the lower and upper bounds of the transformed data.

        Raises:
            NotImplementedError: Raised when `values` is None, as it is currently required for `MolecularInput`.

        """
        if values is None:
            raise NotImplementedError(
                "`values` is currently required for `MolecularInput`",
            )
        data = self.to_descriptor_encoding(transform_type, values)

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


class CategoricalMolecularInput(CategoricalInput, MolecularInput):  # type: ignore
    type: Literal["CategoricalMolecularInput"] = "CategoricalMolecularInput"  # type: ignore
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
    def valid_transform_types() -> List[Union[AnyMolFeatures, CategoricalEncodingEnum]]:  # type: ignore
        return CategoricalInput.valid_transform_types() + [  # type: ignore
            Fingerprints,
            FingerprintsFragments,
            Fragments,
            MordredDescriptors,
        ]

    def get_bounds(  # type: ignore
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
