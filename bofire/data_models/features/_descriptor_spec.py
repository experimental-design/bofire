"""Shared descriptor spec for the descriptor consumers.

A ``DescriptorSpec`` says *which* descriptor columns a consumer should build for a
feature: static numeric columns already stored on the feature, plus columns
*generated* from structure columns (e.g. SMILES) by molecular generators. It is
mixed into the two reductions that consume descriptors:

- ``DescriptorEncoding`` (categorical: one descriptor row per category), and
- ``WeightedSumFeature`` (continuous: one row per component feature).

Correlation-based decorrelation lives here (opt-in via ``filter_descriptors``) and is
applied across the *whole* assembled block — static and generated columns together —
as a pure function, so there is no mutable state on the generators.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import pandas as pd
from pydantic import Field, PrivateAttr, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved
from bofire.data_models.molfeatures.api import AnyMolFeatures


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
        columns: static numeric descriptor columns to use. ``None`` means all
            non-reserved (role ``"descriptor"``) columns of the feature; ``[]`` means
            no static columns (generators only).
        generators: maps a structure column (e.g. ``"smiles"``) to the molecular
            generators run on it. Multiple generators per column are concatenated.
        filter_descriptors: if True, drop correlated columns across the whole block.
        correlation_cutoff: absolute-correlation threshold for filtering.
    """

    columns: Optional[List[str]] = None
    generators: Dict[str, List[AnyMolFeatures]] = Field(default_factory=dict)
    filter_descriptors: bool = False
    correlation_cutoff: float = 0.95

    # frozen post-filter column names; set the first time a table is assembled (or by
    # a consumer's validation) so width is stable and generators stay unmutated.
    _resolved_names: Optional[List[str]] = PrivateAttr(None)

    @model_validator(mode="after")
    def _validate_structure_kind(self):
        """Each generator's ``reads`` must match the kind of its structure column."""
        for column, generators in self.generators.items():
            if not is_reserved(column):
                continue
            reserved = get_reserved_descriptor(column)
            if reserved.role != "structure":
                raise ValueError(
                    f"column '{column}' is not a structure identifier and cannot carry "
                    "descriptor generators.",
                )
            for generator in generators:
                if reserved.kind != generator.reads:
                    raise ValueError(
                        f"generator reads '{generator.reads}' but structure column "
                        f"'{column}' holds '{reserved.kind}'.",
                    )
        return self

    # -- column resolution ---------------------------------------------------------

    def _static_columns(self, feature: "DescriptorsMixin") -> List[str]:
        available = feature.descriptor_columns(role="descriptor")
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
            for generators in self.generators.values()
            for generator in generators
            for name in generator.get_descriptor_names()
        ]

    def declared_names(
        self, feature: Optional["DescriptorsMixin"] = None
    ) -> Optional[List[str]]:
        """Unfiltered descriptor names, or ``None`` if not determinable feature-free.

        Static names need the feature when ``columns is None``; generator names are
        always known. Returns ``None`` only when ``columns is None`` and no feature is
        given (the caller must then resolve against a feature).
        """
        if self.columns is not None:
            static = list(self.columns)
        elif feature is not None:
            static = feature.descriptor_columns(role="descriptor")
        else:
            return None
        return self._check_unique(static + self._generated_names())

    def _check_unique(self, names: List[str]) -> List[str]:
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate descriptor names in descriptor spec: {duplicates}.",
            )
        return names

    # -- table assembly ------------------------------------------------------------

    def _finalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        self._check_unique(list(raw.columns))
        table = (
            filter_correlated(raw, self.correlation_cutoff)
            if (self.filter_descriptors)
            else raw
        )
        self._resolved_names = list(table.columns)
        return table

    def table(self, feature: "DescriptorsMixin") -> pd.DataFrame:
        """Per-level descriptor table for one feature (categorical select-row scope).

        Index = the feature's levels; columns = static columns + generated columns.
        """
        index = feature.descriptor_levels()
        frames: List[pd.DataFrame] = []
        static_cols = self._static_columns(feature)
        if static_cols:
            frames.append(feature.descriptor_table(static_cols))
        for column, generators in self.generators.items():
            structures = pd.Series(self._structure_column(feature, column))
            for generator in generators:
                gen_df = generator.get_descriptor_values(structures)
                gen_df.index = index
                gen_df.columns = generator.get_descriptor_names()
                frames.append(gen_df)
        raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)
        return self._finalize(raw)

    def component_table(self, features: Sequence["DescriptorsMixin"]) -> pd.DataFrame:
        """One descriptor row per component feature (continuous weighted-sum scope).

        Index = component keys. Generators filter once over the combined component
        structures so every row shares the same (filtered) columns.
        """
        index = [feature.key for feature in features]
        frames: List[pd.DataFrame] = []
        static_cols = self._static_columns(features[0])
        if static_cols:
            frames.append(
                pd.concat([f.descriptor_table(static_cols) for f in features])
            )
        for column, generators in self.generators.items():
            structures = pd.Series(
                [self._structure_column(f, column)[0] for f in features]
            )
            for generator in generators:
                gen_df = generator.get_descriptor_values(structures)
                gen_df.index = index
                gen_df.columns = generator.get_descriptor_names()
                frames.append(gen_df)
        raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)
        return self._finalize(raw)

    # -- validation ----------------------------------------------------------------

    def _structure_column(self, feature: "DescriptorsMixin", column: str) -> List[str]:
        if column not in feature.descriptors:
            raise ValueError(
                f"{feature.key}: structure column '{column}' is not present in the "
                "feature's descriptors.",
            )
        return [str(s) for s in feature.descriptors[column]]

    def validate_for(self, feature: "DescriptorsMixin") -> None:
        """Validate ``feature`` carries the data this spec needs (no generation)."""
        static_cols = self._static_columns(feature)
        for column in self.generators:
            self._structure_column(feature, column)
        if not static_cols and not self.generators:
            raise ValueError(
                f"{feature.key}: descriptor spec produces no columns (no static "
                "descriptor columns and no generators).",
            )
