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
from typing import List, Optional

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.descriptor_generators.api import AnyDescriptorGenerator
from bofire.data_models.features.descriptors import Descriptors


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

        Pure: nothing is cached, so rebuilding always yields the same columns. That is what
        lets ``encode`` and ``decode`` assemble the table independently and still agree.
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
        """Ask every compatibility question about ``descriptors`` and this spec.

        This is the gate: it is the only place that copes with a feature carrying no
        block at all, and the only place that raises for an incompatible one. Everything
        below it takes a `Descriptors` and assumes these checks have passed — which is
        why `build` asserts rather than re-checking. It is reached from every entry point
        that can lead to `build`: `Inputs._get_transform_info` / `transform` /
        `inverse_transform` / `get_bounds` via `_validate_transform_specs`, and
        `BotorchSurrogate` at construction.

        Pure metadata — no descriptors are generated here, so this stays usable without
        rdkit. ``key`` only names the offending feature in the messages.
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
