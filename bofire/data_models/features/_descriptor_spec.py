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

    def _static_columns(self, descriptors: Optional[Descriptors]) -> List[str]:
        available = descriptors.names if descriptors is not None else []
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

    def column_names(self, descriptors: Optional[Descriptors]) -> List[str]:
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

    # -- table assembly (shared core) ----------------------------------------------

    def _generated_frames(
        self, structures: pd.Series, index: List
    ) -> List[pd.DataFrame]:
        """Run every generator over ``structures`` and return one frame per generator.

        Args:
            structures: The structure identifiers (SMILES) to generate descriptors
                from — one per row of the table being assembled, in ``index`` order.
                For a categorical that is one SMILES per category; for a weighted sum
                one SMILES per component feature.
            index: The row labels to stamp onto each generated frame (categories, or
                component keys), so the generated columns align with the static ones
                when they are concatenated.

        Returns:
            One frame per generator, each indexed by ``index`` with that generator's
            descriptor names as columns (e.g. ``fingerprint_0 … fingerprint_n``).
        """
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

        This is the shared core of the two scope-specific builders, which differ *only*
        in how they prepare the three arguments:

        - ``DescriptorEncoding.table`` (categorical scope) — one row per **category**,
          because encoding a category means selecting its descriptor row;
        - ``WeightedSumFeature.component_table`` (continuous scope) — one row per
          **component feature**, because the model blends component rows by amount
          (``Σᵢ amountᵢ · rowᵢ``).

        Args:
            index: Row labels of the assembled table — the feature's categories
                (categorical) or the component feature keys (continuous). Also used to
                align the generated frames with the static ones.
            static: The static numeric descriptor columns already stored on the
                feature(s), indexed by ``index``, or None when the spec declares no
                static columns (``columns=[]``, i.e. generators only).
            structures: The SMILES to run the generators over, one per row of ``index``,
                or None when the spec declares no generators.

        Returns:
            The descriptor table: ``index`` as rows, static columns followed by
            generated columns. When ``filter_descriptors`` is set, correlated and
            zero-variance columns are dropped from the *combined* block first.

        Example:
            A categorical solvent with two static columns and 8-bit fingerprints
            (``DescriptorEncoding(columns=["logP", "MW"], generators=[Fingerprints(n_bits=8)])``)
            assembles ``index=["water", "ethanol", "thf"]``, the 3x2 ``static`` frame,
            and ``structures=["O", "CCO", "C1CCOC1"]`` into::

                         logP    MW  fingerprint_0  ...  fingerprint_7
                water    -1.4  18.0            0.0  ...            0.0
                ethanol  -0.3  46.0            1.0  ...            1.0
                thf       0.5  72.0            1.0  ...            1.0

            The same spec on a mixture of three ``ContinuousInput`` components would
            instead be given ``index=["ethanol", "water", "thf"]`` (one row per
            component) and produce a table of the same shape with those row labels.

        Note:
            Pure — correlation filtering happens here when enabled, but no state is
            written, so the same inputs always yield the same columns. That is what
            lets ``encode`` and ``decode`` rebuild the table independently and still
            agree on the filtered column set.
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

    def _structure(self, descriptors: Optional[Descriptors]) -> List[str]:
        if descriptors is None or descriptors.structure is None:
            raise ValueError(
                "has no `structure` column, but the descriptor spec declares "
                "generators that need one.",
            )
        return [str(value) for value in descriptors.structure]

    def validate_for(self, descriptors: Optional[Descriptors], key: str) -> None:
        """Validate that ``descriptors`` carries the data this spec needs.

        Pure metadata: no generation happens here. ``key`` is only used to name the
        offending feature in error messages.
        """
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
