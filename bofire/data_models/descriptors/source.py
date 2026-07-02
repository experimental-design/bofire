"""Base class for descriptor sources.

A ``DescriptorSource`` yields a per-*level* descriptor table for a feature (level =
category / discrete value / the single continuous component). It is the *descriptor
source* axis, orthogonal to the *reduction* (categorical select-row via
``DescriptorEncoding``, continuous amount-weighted-sum via ``WeightedSumFeature``).
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import pandas as pd

from bofire.data_models.base import BaseModel


if TYPE_CHECKING:
    from bofire.data_models.features._descriptors import DescriptorsMixin


class DescriptorSource(BaseModel):
    """Base class for all descriptor sources."""

    type: Any

    @abstractmethod
    def names(self, feature: "DescriptorsMixin") -> List[str]:
        """Unprefixed descriptor column names produced for ``feature``."""

    @abstractmethod
    def declared_names(self) -> Optional[List[str]]:
        """Descriptor names known without a feature, or ``None`` if feature-dependent.

        Used for ``n_transformed_inputs``; for generated sources this reflects the
        generator's *current* (possibly correlation-filtered) descriptor set.
        """

    @abstractmethod
    def check(self, feature: "DescriptorsMixin") -> None:
        """Validate ``feature`` carries the data this source needs (no generation)."""

    @abstractmethod
    def prepare(self, structures: pd.Series) -> None:
        """Optionally fit the source to a set of structures.

        Generated sources use this to run correlation filtering over the given
        structure set; static sources are a no-op.
        """

    @abstractmethod
    def table(self, feature: "DescriptorsMixin") -> pd.DataFrame:
        """Per-level descriptor table (index = feature levels, columns = ``names``)."""

    @abstractmethod
    def component_table(self, features: Sequence["DescriptorsMixin"]) -> pd.DataFrame:
        """One descriptor row per component feature (index = feature keys).

        Unlike :meth:`table` (per-level rows for a *single* feature, the categorical
        select-row scope), this stacks a single row per feature across *many*
        components — the continuous amount-weighted-sum scope. Generated sources
        correlation-filter once over the *combined* component structures so every
        row shares the same descriptor columns.
        """
