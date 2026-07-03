"""Shared per-feature descriptor data (categorical / continuous / discrete).

A feature carries two kinds of per-level data:

- ``descriptors`` — numeric property columns, used directly by encoders/engineered
  features, and
- ``structure`` — an optional column of structure identifiers (SMILES), fed to
  descriptor *generators* (fingerprints, fragments, Mordred) on the surrogate side.

Structures are an explicit field rather than a magic key inside ``descriptors``, so the
distinction is visible in the schema (only SMILES is supported today; the field can be
widened to other structure languages later without breaking callers).
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel


def _validate_smiles(values: List[str]) -> None:
    """Validate that each entry is a parseable SMILES.

    No-op (with a warning) when rdkit is not available, so ``data_models`` stays usable
    without rdkit. Imported lazily to keep this module rdkit-light.
    """
    from bofire.utils.cheminformatics import smiles2mol

    try:
        smiles2mol(values[0])
    except NameError:
        warnings.warn("rdkit not installed, smiles structures cannot be validated.")
        return
    for value in values:
        smiles2mol(value)


class DescriptorsMixin(BaseModel):
    """Mixin giving a feature per-level numeric ``descriptors`` and an optional
    ``structure`` column.

    A "level" is a category (categorical), a discrete value (discrete), or the single
    component (continuous). Each ``descriptors`` column and the ``structure`` column
    (when set) has one entry per level.

    Subclasses must implement :meth:`descriptor_levels`.
    """

    if TYPE_CHECKING:
        # always mixed into a ``Feature`` (which supplies ``key``); declared here
        # only so the descriptor helpers and specs can reference ``self.key``.
        key: str

    descriptors: Dict[str, List[float]] = Field(default_factory=dict)
    structure: Optional[List[str]] = None

    def descriptor_levels(self) -> List:
        """The row labels of the descriptor table.

        A numeric feature (continuous / discrete) is a single component — one
        descriptor row, keyed by the feature. ``CategoricalInput`` overrides this with
        one row per category.
        """
        return [self.key]

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_smiles(cls, data):
        """Back-compat: old dumps stored SMILES as a ``"smiles"`` descriptor column."""
        if isinstance(data, dict) and isinstance(data.get("descriptors"), dict):
            descriptors = dict(data["descriptors"])
            if "smiles" in descriptors and data.get("structure") is None:
                data = {**data, "structure": descriptors.pop("smiles")}
                data["descriptors"] = descriptors
        return data

    @field_validator("descriptors")
    @classmethod
    def _coerce_descriptors(cls, descriptors):
        """Coerce every descriptor column to numeric. Length is checked against the
        levels in the model validator below."""
        validated: Dict[str, List[float]] = {}
        for name, column in descriptors.items():
            try:
                validated[name] = [float(v) for v in column]
            except (TypeError, ValueError):
                raise ValueError(f"descriptor column '{name}' must be numeric")
        return validated

    @field_validator("structure")
    @classmethod
    def _validate_structure(cls, structure):
        if structure is not None:
            _validate_smiles([str(s) for s in structure])
        return structure

    @model_validator(mode="after")
    def _validate_descriptor_lengths(self):
        n = len(self.descriptor_levels())
        for name, column in self.descriptors.items():
            if len(column) != n:
                raise ValueError(
                    f"descriptor column '{name}' must have {n} value(s) (one per level)",
                )
        if self.structure is not None and len(self.structure) != n:
            raise ValueError(f"structure must have {n} value(s) (one per level)")
        return self

    def descriptor_columns(self) -> List[str]:
        """Names of the numeric descriptor columns."""
        return list(self.descriptors.keys())

    def descriptor_table(self, columns: List[str]) -> pd.DataFrame:
        """Per-level table (rows = levels, columns = selected descriptors)."""
        return pd.DataFrame(
            {c: self.descriptors[c] for c in columns},
            index=self.descriptor_levels(),
        )
