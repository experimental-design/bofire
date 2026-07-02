import warnings
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import field_validator, model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
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
