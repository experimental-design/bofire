import warnings
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import field_validator, model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.utils.cheminformatics import smiles2mol


class ContinuousMolecularInput(ContinuousInput):
    """Deprecated. Use :class:`ContinuousInput` with a ``smiles`` descriptor column
    instead.

    Kept as a thin deserialization shim: the single ``molecule`` SMILES is mirrored
    into a reserved ``smiles`` descriptor column.
    """

    type: Literal["ContinuousMolecularInput"] = "ContinuousMolecularInput"
    order_id: ClassVar[int] = 4

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_molecule(cls, data):
        if not isinstance(data, dict):
            return data
        warnings.warn(
            "`ContinuousMolecularInput` is deprecated, use `ContinuousInput` with "
            "a `smiles` descriptor column instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        molecule = data.pop("molecule", None)
        descriptors = dict(data.get("descriptors") or {})
        if molecule is not None and "smiles" not in descriptors:
            descriptors["smiles"] = [molecule]
            data["descriptors"] = descriptors
        return data


class CategoricalMolecularInput(CategoricalInput):
    """Deprecated. Use :class:`CategoricalInput` with a ``smiles`` descriptor column
    and a :class:`DescriptorEncoding` on the surrogate instead.

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
            "a `smiles` descriptor column and a `DescriptorEncoding` instead.",
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
