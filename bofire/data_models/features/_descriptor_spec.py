"""Shared descriptor spec for the descriptor consumers.

A ``DescriptorSpec`` says *which* descriptor columns a consumer should build for a
feature: static numeric columns already stored on the feature (``columns``), plus columns
*generated* from the feature's ``structure`` (SMILES) by descriptor generators
(``generators``). It is mixed into the two reductions that consume descriptors:

- ``DescriptorEncoding`` (categorical: one descriptor row per category), and
- ``WeightedSumFeature`` (continuous: one row per component feature).

Both scopes reduce to "one descriptor block plus row labels", so the spec exposes a single
``build(descriptors, index)`` and a single ``resolved_names(descriptors, index)``. Each
consumer only has to say what its rows are: a categorical passes its own block and its
categories, a weighted sum stacks its components' blocks with ``Descriptors.concat`` and
passes their keys.

Correlation-based decorrelation (opt-in via ``filter_descriptors``) is applied across the
*whole* assembled block — static and generated columns together — as a pure function, so
there is no mutable state on the generators or the spec.
"""

from collections import Counter
from typing import Annotated, List, Optional

import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.descriptor_generators.api import AnyDescriptorGenerator
from bofire.data_models.features.descriptors import Descriptors


def filter_correlated(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Drop zero-variance columns, then greedily drop columns with ``|corr| > cutoff``.

    The first column of each correlated group is kept, so column order decides ties: put
    static (interpretable) columns ahead of generated ones to keep them.
    """
    variances = df.var()
    non_constant = variances[variances > 0].index.tolist()
    if len(non_constant) == 0:
        raise ValueError(
            "No descriptors with non-zero variance found. "
            "Cannot perform correlation-based filtering.",
        )
    df = df[non_constant]
    if df.shape[1] == 1:
        return df
    correlation = df.corr().abs()
    selected: List[str] = []
    remaining = set(range(len(df.columns)))
    while remaining:
        current = min(remaining)
        selected.append(df.columns[current])
        remaining.remove(current)
        remaining -= {
            idx for idx in remaining if correlation.iloc[current, idx] > cutoff
        }
    return df[selected]


class DescriptorSpec(BaseModel):
    """Mixin declaring how to build a descriptor table for a feature."""

    columns: Optional[List[str]] = Field(
        default=None,
        description="Which of the feature's stored numeric descriptor columns to use. "
        "Null means all of them; an empty list means none, leaving only the generated "
        "columns.",
    )
    generators: List[AnyDescriptorGenerator] = Field(
        default=[],
        description="Descriptor generators run on the feature's SMILES structures, "
        "such as fingerprints or Mordred descriptors. Their outputs are concatenated "
        "onto the static columns. Empty means nothing is generated.",
    )
    filter_descriptors: bool = Field(
        default=False,
        description="Whether to drop correlated columns across the assembled block, "
        "static and generated together. Useful when generators produce hundreds of "
        "near-duplicate columns.",
    )
    correlation_cutoff: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.95,
        description="Absolute correlation above which a column is dropped when "
        "`filter_descriptors` is enabled. Of each correlated group the first column is "
        "kept, so static columns listed before generated ones survive.",
    )

    @model_validator(mode="after")
    def validate_declared_names_unique(self):
        """The names this spec declares by itself must be unique.

        Self-consistency only: with ``columns=None`` the feature's own columns are unknown
        here, so a collision between one of those and a generated name is the gate's
        business (see :meth:`validate_for`).

        Skipped unless something can actually collide -- each generator already guarantees
        its own names are unique, so one generator and no listed columns cannot conflict.
        Worth the guard: generating the 2048 names of a default ``Fingerprints`` costs
        ~0.75 ms against ~0.003 ms for the construction itself, and the default encoding
        takes exactly that shape.
        """
        if len(self.generators) < 2 and not self.columns:
            return self
        names = list(self.columns or []) + self._generated_names()
        duplicates = sorted(n for n, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"descriptor names must be unique, got duplicates {duplicates}. Two "
                "generators producing the same names, a column listed twice, or a column "
                "colliding with a generated one will do this.",
            )
        return self

    # -- column resolution ---------------------------------------------------------

    def _static_columns(self, descriptors: Descriptors) -> List[str]:
        """The static columns this spec selects; ``columns=None`` means all of them."""
        return descriptors.names if self.columns is None else list(self.columns)

    def _generated_names(self) -> List[str]:
        """The columns the generators declare, in the order ``build`` appends them."""
        return [
            name
            for generator in self.generators
            for name in generator.get_descriptor_names()
        ]

    def column_names(self, descriptors: Descriptors) -> List[str]:
        """Declared (pre-filter) descriptor column names for a descriptor block.

        Static columns resolved against the block, followed by generator columns.
        Uniqueness is the gate's business, see :meth:`validate_for`.
        """
        return self._static_columns(descriptors) + self._generated_names()

    def resolved_names(self, descriptors: Descriptors, index: List) -> List[str]:
        """The column names a consumer actually produces for a descriptor block.

        The same as :meth:`column_names` unless ``filter_descriptors`` is set, which can
        only be resolved by assembling the table — filtering drops columns by *value*.
        Assembling runs the generators, and therefore needs rdkit, so the declared names
        are used whenever filtering is off.

        Args:
            descriptors: The block the consumer reads.
            index: Row labels, only used when the table has to be assembled.
        """
        if self.filter_descriptors:
            return list(self.build(descriptors, index).columns)
        return self.column_names(descriptors)

    # -- table assembly --------------------------------------------------------------

    def build(self, descriptors: Descriptors, index: List) -> pd.DataFrame:
        """Build the descriptor table: static columns ‖ generated columns.

        Args:
            descriptors: The block to read static columns and structures from.
            index: Row labels for the result (categories, or component keys).

        Returns:
            ``index`` as rows; the selected static columns followed by every generator's
            columns. With ``filter_descriptors`` set, zero-variance and correlated columns
            are dropped from the *combined* block.

        Deterministic and uncached: two calls with the same arguments yield the same
        columns, so independent callers agree without sharing state.
        """
        frames: List[pd.DataFrame] = []

        static_columns = self._static_columns(descriptors)
        if static_columns:
            frames.append(descriptors.table(index, static_columns))

        if self.generators:
            assert descriptors.structure is not None  # guaranteed by validate_for
            structures = pd.Series(descriptors.structure)
            for generator in self.generators:
                generated = generator.get_descriptor_values(structures)
                generated.index = index
                generated.columns = generator.get_descriptor_names()
                frames.append(generated)

        assert frames  # validate_for rejects a spec that produces no columns
        raw = pd.concat(frames, axis=1)
        return (
            filter_correlated(raw, self.correlation_cutoff)
            if self.filter_descriptors
            else raw
        )

    # -- validation ----------------------------------------------------------------

    def validate_for(self, descriptors: Optional[Descriptors], key: str) -> None:
        """Check that ``descriptors`` can satisfy this spec.

        Reads metadata only: no descriptors are generated, so this is usable without
        rdkit. Callers may treat a block that passes as buildable.

        Args:
            descriptors: The block to check, or None if the feature carries none.
            key: Names the offending feature in the error messages.

        Raises:
            ValueError: If there is no block; if a column listed in ``columns`` is not
                among the block's numeric descriptors; if ``generators`` are declared but
                the block has no ``structure`` to run them on; if the resulting column
                names are not unique; or if the spec would produce no columns at all.
        """
        if descriptors is None:
            raise ValueError(f"{key}: carries no descriptors for this spec to read.")

        available = descriptors.names
        missing = [c for c in self.columns or [] if c not in available]
        if missing:
            raise ValueError(
                f"{key}: descriptor columns {missing} are not available as numeric "
                f"descriptors. Available: {sorted(available)}.",
            )

        if self.generators and descriptors.structure is None:
            raise ValueError(
                f"{key}: the descriptor spec declares generators, but no `structure` "
                "column is available to run them on.",
            )

        names = self.column_names(descriptors)
        duplicates = sorted(n for n, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"{key}: descriptor names must be unique, got duplicates {duplicates}. "
                "Two generators producing the same names, or a static column colliding "
                "with a generated one, will do this.",
            )

        if not names:
            raise ValueError(
                f"{key}: descriptor spec produces no columns (no static descriptor "
                "columns and no generators).",
            )
