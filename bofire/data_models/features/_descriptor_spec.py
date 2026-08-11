"""Shared descriptor spec for the descriptor consumers.

A ``DescriptorSpec`` says *which* descriptor columns a consumer should build for a
feature: static numeric columns already stored on the feature (``columns``), plus columns
*generated* from the feature's ``structure`` (SMILES) by descriptor generators
(``generators``). It is mixed into the two reductions that consume descriptors:

- ``DescriptorEncoding`` (categorical: one descriptor row per category), and
- ``WeightedSumFeature`` (continuous: one row per component feature).

Both scopes reduce to "one descriptor block plus row labels", so the spec exposes a single
``build(descriptors, index)``. Each consumer only has to say what its rows are: a
categorical passes its own block and its categories, a weighted sum stacks its components'
blocks with ``Descriptors.concat`` and passes their keys.

Correlation-based decorrelation (opt-in via ``filter_descriptors``) is applied across the
*whole* assembled block — static and generated columns together — as a pure function, so
there is no mutable state on the generators or the spec.
"""

from typing import List, Optional

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.descriptor_generators.api import AnyDescriptorGenerator
from bofire.data_models.features._descriptors import Descriptors


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

    def _static_columns(self, descriptors: Descriptors) -> List[str]:
        available = descriptors.names
        if self.columns is None:
            return available
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise ValueError(
                f"descriptor columns {missing} are not available as numeric "
                f"descriptors. Available: {sorted(available)}.",
            )
        return list(self.columns)

    def _generated_names(self) -> List[str]:
        return [
            name
            for generator in self.generators
            for name in generator.get_descriptor_names()
        ]

    def column_names(self, descriptors: Descriptors) -> List[str]:
        """Declared (pre-filter) descriptor column names for a descriptor block.

        Static columns resolved against the block, followed by generator columns. The
        post-filter names, when ``filter_descriptors`` is set, are the columns of an
        assembled table.
        """
        return self._check_unique(
            self._static_columns(descriptors) + self._generated_names()
        )

    def _check_unique(self, names: List[str]) -> List[str]:
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate descriptor names in descriptor spec: {duplicates}.",
            )
        return names

    # -- table assembly --------------------------------------------------------------

    def build(self, descriptors: Descriptors, index: List) -> pd.DataFrame:
        """Build the descriptor table: static columns ‖ generated columns.

        Both descriptor scopes reduce to "one block plus row labels":

        - categorical — the feature's block, one row per category
          (:meth:`DescriptorEncoding.table`);
        - continuous — the components' one-row blocks stacked with
          :meth:`Descriptors.concat`, one row per component
          (:meth:`WeightedSumFeature.component_table`).

        Args:
            descriptors: The block to read static columns and structures from.
            index: Row labels for the result (categories, or component keys).

        Returns:
            ``index`` as rows; the selected static columns followed by every generator's
            columns. With ``filter_descriptors`` set, zero-variance and correlated columns
            are dropped from the *combined* block.

        Pure: nothing is cached, so rebuilding always yields the same columns. That is what
        lets ``encode`` and ``decode`` assemble the table independently and still agree.
        """
        frames: List[pd.DataFrame] = []

        static_columns = self._static_columns(descriptors)
        if static_columns:
            frames.append(descriptors.table(index, static_columns))

        if self.generators:
            structures = pd.Series(self._structure(descriptors))
            for generator in self.generators:
                generated = generator.get_descriptor_values(structures)
                generated.index = index
                generated.columns = generator.get_descriptor_names()
                frames.append(generated)

        raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)
        self._check_unique(list(raw.columns))
        return (
            filter_correlated(raw, self.correlation_cutoff)
            if self.filter_descriptors
            else raw
        )

    # -- validation ----------------------------------------------------------------

    def _structure(self, descriptors: Descriptors) -> List[str]:
        if descriptors.structure is None:
            raise ValueError(
                "the descriptor spec declares generators, but no `structure` column "
                "is available to run them on.",
            )
        return [str(value) for value in descriptors.structure]

    def validate_for(self, descriptors: Optional[Descriptors], key: str) -> None:
        """Validate that ``descriptors`` carries the data this spec needs.

        This is the *only* place that has to cope with a feature carrying no block at
        all; everything below it takes a `Descriptors` and assumes this gate has run. It
        is reached from every entry point that can lead to `build` —
        `Inputs._get_transform_info` / `transform` / `inverse_transform` / `get_bounds`
        via `_validate_transform_specs`, and `BotorchSurrogate` at construction.

        Pure metadata: no generation happens here. ``key`` is only used to name the
        offending feature in error messages.
        """
        if descriptors is None:
            raise ValueError(f"{key}: carries no descriptors for this spec to read.")
        try:
            static_cols = self._static_columns(descriptors)
            if self.generators:
                self._structure(descriptors)
        except ValueError as err:
            raise ValueError(f"{key}: {err}") from err
        if not static_cols and not self.generators:
            raise ValueError(
                f"{key}: descriptor spec produces no columns (no static descriptor "
                "columns and no generators).",
            )
