"""Shared descriptor spec for the descriptor consumers.

A ``DescriptorSpec`` says *which* descriptor columns a consumer should build for a
feature: static numeric columns already stored on the feature (``columns``), plus columns
*generated* from the feature's ``structure`` (SMILES) by descriptor generators
(``generators``). It is mixed into the two reductions that consume descriptors:

- ``DescriptorEncoding`` (categorical: one descriptor row per category), and
- ``WeightedSumFeature`` (continuous: one row per component feature).

This class owns the *scope-agnostic* machinery: column-name resolution, correlation
decorrelation, and the shared ``_assemble`` core. The scope-specific shaping — what a
row *is* — lives on each consumer as ``table`` / ``component_table``, which just prepare
``(index, static, structures)`` and delegate here.

Correlation-based decorrelation (opt-in via ``filter_descriptors``) is applied across the
*whole* assembled block — static and generated columns together — as a pure function, so
there is no mutable state on the generators or the spec.
"""

from typing import TYPE_CHECKING, List, Optional

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.descriptor_generators.api import AnyDescriptorGenerator


if TYPE_CHECKING:
    from bofire.data_models.features._descriptors import DescriptorsMixin


def filter_correlated(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Drop zero-variance columns, then greedily drop columns with ``|corr| > cutoff``.

    Pure: depends only on ``df``. The first column of each correlated group is kept, so
    ordering the frame with static (interpretable) columns first preserves them on ties.
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
    """Mixin declaring how to build a descriptor table for a feature.

    Attributes:
        columns: static numeric descriptor columns to use. ``None`` means all of the
            feature's numeric descriptor columns; ``[]`` means none (generators only).
        generators: descriptor generators run on the feature's ``structure`` column;
            their outputs are concatenated. Empty means no generated columns.
        filter_descriptors: if True, drop correlated columns across the whole block.
        correlation_cutoff: absolute-correlation threshold for filtering.
    """

    columns: Optional[List[str]] = None
    generators: List[AnyDescriptorGenerator] = Field(default_factory=list)
    filter_descriptors: bool = False
    correlation_cutoff: float = 0.95

    # -- column resolution ---------------------------------------------------------

    def _static_columns(self, feature: "DescriptorsMixin") -> List[str]:
        available = feature.descriptor_columns()
        if self.columns is None:
            return available
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise ValueError(
                f"{feature.key}: descriptor columns {missing} are not available as "
                f"numeric descriptors. Available: {sorted(available)}.",
            )
        return list(self.columns)

    def _generated_names(self) -> List[str]:
        return [
            name
            for generator in self.generators
            for name in generator.get_descriptor_names()
        ]

    def column_names(self, feature: "DescriptorsMixin") -> List[str]:
        """Declared (pre-filter) descriptor column names for a feature.

        Static columns resolved against the feature, followed by generator columns.
        Always returns a list (a feature is required); post-filter names, when
        ``filter_descriptors`` is set, are the columns of an assembled table.
        """
        return self._check_unique(
            self._static_columns(feature) + self._generated_names()
        )

    def _check_unique(self, names: List[str]) -> List[str]:
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate descriptor names in descriptor spec: {duplicates}.",
            )
        return names

    # -- table assembly (shared core) ----------------------------------------------

    def _generated_frames(
        self, structures: pd.Series, index: List
    ) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for generator in self.generators:
            gen_df = generator.get_descriptor_values(structures)
            gen_df.index = index
            gen_df.columns = generator.get_descriptor_names()
            frames.append(gen_df)
        return frames

    def _assemble(
        self,
        index: List,
        static: Optional[pd.DataFrame],
        structures: Optional[pd.Series],
    ) -> pd.DataFrame:
        """Assemble a descriptor table from prepared parts (static ‖ generated).

        Shared by the two scope-specific builders — ``DescriptorEncoding.table`` (one
        row per category) and ``WeightedSumFeature.component_table`` (one row per
        component) — which differ only in how they prepare ``index`` / ``static`` /
        ``structures``. Pure: correlation filtering is applied here when enabled, but
        no state is written, so the same inputs always yield the same columns.
        """
        frames: List[pd.DataFrame] = []
        if static is not None:
            frames.append(static)
        if structures is not None:
            frames += self._generated_frames(structures, index)
        raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)
        self._check_unique(list(raw.columns))
        return (
            filter_correlated(raw, self.correlation_cutoff)
            if self.filter_descriptors
            else raw
        )

    # -- validation ----------------------------------------------------------------

    def _structure(self, feature: "DescriptorsMixin") -> List[str]:
        if feature.structure is None:
            raise ValueError(
                f"{feature.key}: has no `structure` column, but the descriptor spec "
                "declares generators that need one.",
            )
        return [str(s) for s in feature.structure]

    def validate_for(self, feature: "DescriptorsMixin") -> None:
        """Validate ``feature`` carries the data this spec needs (no generation)."""
        static_cols = self._static_columns(feature)
        if self.generators:
            self._structure(feature)
        if not static_cols and not self.generators:
            raise ValueError(
                f"{feature.key}: descriptor spec produces no columns (no static "
                "descriptor columns and no generators).",
            )
