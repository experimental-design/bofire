"""Composite descriptor source: concatenate several sources (static + molecular)."""

from typing import TYPE_CHECKING, Annotated, List, Literal, Optional, Sequence, Union

import pandas as pd
from pydantic import Field

from bofire.data_models.descriptors.generated import GeneratedSource
from bofire.data_models.descriptors.source import DescriptorSource
from bofire.data_models.descriptors.static import StaticSource


if TYPE_CHECKING:
    from bofire.data_models.features._descriptors import DescriptorsMixin


class CompositeSource(DescriptorSource):
    """Concatenate several descriptor sources into one column space.

    This is what makes handcrafted + molecular descriptors combinable on one
    feature. Descriptor names across components must be unique.
    """

    type: Literal["CompositeSource"] = "CompositeSource"
    sources: Annotated[List[Union[StaticSource, GeneratedSource]], Field(min_length=2)]

    def _check_unique(self, names: List[str]) -> List[str]:
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate descriptor names in CompositeSource: {duplicates}",
            )
        return names

    def prepare(self, structures: pd.Series) -> None:
        for source in self.sources:
            source.prepare(structures)

    def names(self, feature: "DescriptorsMixin") -> List[str]:
        names: List[str] = []
        for source in self.sources:
            names += source.names(feature)
        return self._check_unique(names)

    def declared_names(self) -> Optional[List[str]]:
        names: List[str] = []
        for source in self.sources:
            declared = source.declared_names()
            if declared is None:
                return None
            names += declared
        return self._check_unique(names)

    def check(self, feature: "DescriptorsMixin") -> None:
        for source in self.sources:
            source.check(feature)

    def table(self, feature: "DescriptorsMixin") -> pd.DataFrame:
        combined = pd.concat([source.table(feature) for source in self.sources], axis=1)
        self._check_unique(list(combined.columns))
        return combined

    def component_table(self, features: Sequence["DescriptorsMixin"]) -> pd.DataFrame:
        combined = pd.concat(
            [source.component_table(features) for source in self.sources], axis=1
        )
        self._check_unique(list(combined.columns))
        return combined
