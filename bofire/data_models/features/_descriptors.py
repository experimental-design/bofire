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

    An empty list is left to the per-level length check, which reports it properly;
    probing it here would raise ``IndexError`` instead of a validation error.
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


class DescriptorsMixin(BaseModel):
    """Mixin giving a feature per-level numeric ``descriptors`` and an optional
    ``structure`` column.

    A "level" is a row of the feature's descriptor table. Every ``descriptors`` column
    and the ``structure`` column (when set) has exactly one entry per level, in the
    order given by :meth:`descriptor_levels`.

    How many levels a feature has depends on its kind:

    ``CategoricalInput`` — **one level per category**. The feature picks the row of the
    chosen category (select-row), so each column needs one value per category::

        CategoricalInput(
            key="solvent",
            categories=["water", "ethanol", "thf"],
            descriptors={"logP": [-1.4, -0.3, 0.5], "MW": [18.0, 46.0, 72.0]},
            structure=["O", "CCO", "C1CCOC1"],
        )
        # descriptor_levels() -> ["water", "ethanol", "thf"]   (3 rows)

    ``ContinuousInput`` — **a single level** (the feature itself). It is one *component*
    of a mixture whose amount weights its descriptor row, so one value per column::

        ContinuousInput(
            key="ethanol",
            bounds=(0, 1),
            descriptors={"logP": [-0.3], "MW": [46.0]},
            structure=["CCO"],
        )
        # descriptor_levels() -> ["ethanol"]                   (1 row)

    ``DiscreteInput`` — **also a single level**, like continuous. A discrete input is one
    numeric quantity that happens to be restricted to a set of allowed values; the
    allowed values are *not* descriptor levels (an amount of ethanol restricted to
    ``[0, 0.5, 1]`` still describes one substance, so it needs one SMILES, not three)::

        DiscreteInput(
            key="ethanol",
            values=[0.0, 0.5, 1.0],
            descriptors={"logP": [-0.3], "MW": [46.0]},
            structure=["CCO"],
        )
        # descriptor_levels() -> ["ethanol"]                   (1 row)

    Subclasses must implement :meth:`descriptor_levels`; the default here is the
    single-component case, which ``CategoricalInput`` overrides with one row per
    category.

    Attributes:
        descriptors: Numeric property columns, one value per level. Defaults to `{}`.
        structure: Optional structure identifiers (SMILES), one per level, fed to
            descriptor *generators* (fingerprints, fragments, Mordred) on the surrogate
            side. Defaults to None.
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

    def has_descriptor_data(self) -> bool:
        """Whether the feature carries descriptor data of any kind.

        True when it has numeric descriptor columns and/or a structure column, i.e. when
        it can be descriptor-encoded rather than treated as a plain set of levels. This
        is a property of the *data*, not of the class.
        """
        return bool(self.descriptors) or self.structure is not None

    def descriptor_table(self, columns: List[str]) -> pd.DataFrame:
        """Per-level table (rows = levels, columns = selected descriptors)."""
        return pd.DataFrame(
            {c: self.descriptors[c] for c in columns},
            index=self.descriptor_levels(),
        )
