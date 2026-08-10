"""Per-level descriptor data, as a value object carried by a feature.

A :class:`Descriptors` block holds two kinds of per-level data:

- ``columns`` — numeric property columns, used directly by encoders/engineered features,
  and
- ``structure`` — an optional column of structure identifiers (SMILES), fed to descriptor
  *generators* (fingerprints, fragments, Mordred) on the surrogate side.

Structures are an explicit field rather than a magic key inside ``columns``, so the
distinction is visible in the schema (only SMILES is supported today; the field can be
widened to other structure languages later without breaking callers).

The block knows nothing about features. It enforces that it is internally consistent —
every column, and the structure, describes the same number of levels — while the *fit*
to a particular feature (one row per category, or per single component) is checked by the
feature itself via :func:`validate_descriptors_fit`.
"""

import warnings
from typing import Dict, List, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel


def _validate_smiles(values: List[str]) -> None:
    """Validate that each entry is a parseable SMILES.

    No-op (with a warning) when rdkit is not available, so ``data_models`` stays usable
    without rdkit. Imported lazily to keep this module rdkit-light.

    An empty list is left to the length checks, which report it properly; probing it here
    would raise ``IndexError`` instead of a validation error.
    """
    from bofire.utils.cheminformatics import smiles2mol

    if not values:
        return
    try:
        smiles2mol(values[0])
    except NameError:
        warnings.warn("rdkit not installed, smiles structures cannot be validated.")
        return
    for value in values:
        smiles2mol(value)


class Descriptors(BaseModel):
    """A rectangular block of per-level descriptor data.

    A "level" is a row of the block. What the levels *are* depends on the feature the
    block is attached to, and is the feature's business, not this class's:

    ``CategoricalInput`` — **one level per category**, so the block picks the row of the
    chosen category (select-row)::

        CategoricalInput(
            key="solvent",
            categories=["water", "ethanol", "thf"],
            descriptors=Descriptors(
                columns={"logP": [-1.4, -0.3, 0.5], "MW": [18.0, 46.0, 72.0]},
                structure=["O", "CCO", "C1CCOC1"],
            ),
        )

    ``ContinuousInput`` / ``DiscreteInput`` — **a single level**, the feature itself. It
    is one *component* of a mixture whose amount weights its row, so each column holds one
    value. For a discrete input the allowed ``values`` are not levels: a restricted amount
    of a substance still describes one substance::

        ContinuousInput(
            key="ethanol",
            bounds=(0, 1),
            descriptors=Descriptors(columns={"logP": [-0.3]}, structure=["CCO"]),
        )

    Attributes:
        columns: Numeric property columns, each with one value per level.
        structure: Optional structure identifiers (SMILES), one per level.
    """

    columns: Dict[str, List[float]] = Field(default_factory=dict)
    structure: Optional[List[str]] = None

    @field_validator("columns")
    @classmethod
    def _coerce_columns(cls, columns):
        """Coerce every column to numeric; lengths are checked in the model validator."""
        validated: Dict[str, List[float]] = {}
        for name, column in columns.items():
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
    def _validate_rectangular(self):
        """Every column and the structure must describe the same levels."""
        lengths = {name: len(column) for name, column in self.columns.items()}
        if self.structure is not None:
            lengths["structure"] = len(self.structure)
        if len(set(lengths.values())) > 1:
            raise ValueError(
                "all descriptor columns and the structure column must have the same "
                f"number of values (one per level), got {lengths}",
            )
        if not lengths:
            raise ValueError(
                "descriptors must declare at least one column or a structure",
            )
        return self

    def __len__(self) -> int:
        """Number of levels this block describes."""
        if self.structure is not None:
            return len(self.structure)
        return len(next(iter(self.columns.values())))

    @property
    def names(self) -> List[str]:
        """Names of the numeric descriptor columns."""
        return list(self.columns)

    def table(self, index: List, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Per-level table (rows = ``index``, columns = ``columns`` or all of them)."""
        selected = self.names if columns is None else columns
        return pd.DataFrame({c: self.columns[c] for c in selected}, index=index)


def validate_descriptors_fit(descriptors: Optional[Descriptors], levels: List) -> None:
    """Check that a block describes exactly the levels of the feature carrying it.

    The block itself only guarantees internal consistency; how many levels there should be
    is the feature's business, so the feature calls this from its own model validator.
    """
    if descriptors is None:
        return
    if len(descriptors) != len(levels):
        raise ValueError(
            f"descriptors must have {len(levels)} value(s) per column (one per level), "
            f"got {len(descriptors)}",
        )
