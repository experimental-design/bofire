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
feature itself via :meth:`Descriptors.validate_fit`.
"""

import warnings
from typing import Dict, List, Optional, Sequence

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

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, structure):
        if structure is not None:
            _validate_smiles(structure)
        return structure

    @model_validator(mode="after")
    def validate_rectangular(self):
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

    def validate_fit(self, levels: List) -> None:
        """Check that this block describes exactly ``levels``.

        The block itself only guarantees internal consistency; how many levels there
        should be is the feature's business, so the feature calls this from its own
        model validator.

        Args:
            levels: The levels the feature carrying this block declares.

        Raises:
            ValueError: If the block describes a different number of levels.
        """
        if len(self) != len(levels):
            raise ValueError(
                f"descriptors must have {len(levels)} value(s) per column (one per "
                f"level), got {len(self)}",
            )

    def table(self, index: List, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Per-level table (rows = ``index``, columns = ``columns`` or all of them)."""
        selected = self.names if columns is None else columns
        return pd.DataFrame({c: self.columns[c] for c in selected}, index=index)

    @classmethod
    def concat(cls, blocks: Sequence[Optional["Descriptors"]]) -> "Descriptors":
        """Stack blocks row-wise into a single block.

        Used to turn the one-row blocks of the components of a mixture into one block the
        weighted sum can read, so that both descriptor scopes end up as "a block plus an
        index". The blocks must describe the same thing: same column names and either all
        or none carrying a structure.

        Column *order* follows the first block, which keeps static columns ahead of
        generated ones — :func:`filter_correlated` keeps the first of each correlated
        group, so the order is load-bearing.
        """
        if not blocks:
            raise ValueError("cannot concatenate an empty list of descriptor blocks")
        missing = [i for i, block in enumerate(blocks) if block is None]
        if missing:
            raise ValueError(
                f"components at positions {missing} carry no descriptors, so they "
                "cannot be combined with ones that do",
            )

        first = blocks[0]
        names = first.names
        for block in blocks[1:]:
            if set(block.names) != set(names):
                raise ValueError(
                    "all components must carry the same descriptor columns, got "
                    f"{sorted(names)} and {sorted(block.names)}",
                )
            if (block.structure is None) != (first.structure is None):
                raise ValueError(
                    "either all components carry a `structure` or none do",
                )

        columns = {name: [v for b in blocks for v in b.columns[name]] for name in names}
        # `or []` never fires — the loop above established that every block agrees with
        # `first` on structure presence — but it keeps the comprehension provably iterable.
        structure = (
            [s for b in blocks for s in (b.structure or [])]
            if first.structure is not None
            else None
        )
        return cls(columns=columns, structure=structure)
