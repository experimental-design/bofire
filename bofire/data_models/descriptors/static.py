"""Static descriptor source: numeric descriptor columns stored on the feature."""

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence

import pandas as pd

from bofire.data_models.descriptors.source import DescriptorSource


if TYPE_CHECKING:
    from bofire.data_models.features._descriptors import DescriptorsMixin


class StaticSource(DescriptorSource):
    """Use (a subset of) the feature's numeric descriptor columns directly.

    Attributes:
        columns: Names of the numeric descriptor columns to use. ``None`` means
            all non-reserved (role ``"descriptor"``) columns in declared order.
    """

    type: Literal["StaticSource"] = "StaticSource"
    columns: Optional[List[str]] = None

    def _columns(self, feature: "DescriptorsMixin") -> List[str]:
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

    def names(self, feature: "DescriptorsMixin") -> List[str]:
        return self._columns(feature)

    def declared_names(self) -> Optional[List[str]]:
        return list(self.columns) if self.columns is not None else None

    def check(self, feature: "DescriptorsMixin") -> None:
        if not self._columns(feature):
            raise ValueError(
                f"{feature.key}: no numeric descriptor columns available for a "
                "static descriptor source.",
            )

    def prepare(self, structures: pd.Series) -> None:
        return None

    def table(self, feature: "DescriptorsMixin") -> pd.DataFrame:
        return feature.descriptor_table(self._columns(feature))

    def component_table(self, features: Sequence["DescriptorsMixin"]) -> pd.DataFrame:
        # each component's static table is a single row indexed by its key.
        return pd.concat([self.table(feature) for feature in features])
