"""Generated descriptor source: descriptors computed from a structure column."""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence

import pandas as pd
from pydantic import PrivateAttr, model_validator

from bofire.data_models.descriptors.source import DescriptorSource
from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved
from bofire.data_models.molfeatures.api import AnyMolFeatures


if TYPE_CHECKING:
    from bofire.data_models.features._descriptors import DescriptorsMixin


class GeneratedSource(DescriptorSource):
    """Generate descriptors by running a molecular generator on a structure column.

    The structures (e.g. SMILES) are *data* on the feature (a reserved structure
    column); the generator and its hyperparameters are the per-surrogate choice and
    live here.

    Attributes:
        structure: Name of the reserved structure column to read.
        generator: The molecular feature generator (Fingerprints/Fragments/Mordred/Composite).
    """

    type: Literal["GeneratedSource"] = "GeneratedSource"
    structure: str = "smiles"
    generator: AnyMolFeatures

    # correlation-filtering result cached per structure-set, so reusing one source
    # across features/scopes never leaks one set's filtered descriptors into another.
    _filtered_cache: Dict[tuple, Optional[List[str]]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_structure_kind(self):
        """The structure column's kind must match what the generator reads."""
        if is_reserved(self.structure):
            reserved = get_reserved_descriptor(self.structure)
            if reserved.role != "structure":
                raise ValueError(
                    f"column '{self.structure}' is not a structure identifier and "
                    "cannot be used for a generated descriptor source.",
                )
            if reserved.kind != self.generator.reads:
                raise ValueError(
                    f"generator reads '{self.generator.reads}' but structure column "
                    f"'{self.structure}' holds '{reserved.kind}'.",
                )
        return self

    def _structure_column(self, feature: "DescriptorsMixin") -> List[str]:
        if self.structure not in feature.descriptors:
            raise ValueError(
                f"{feature.key}: structure column '{self.structure}' is not present "
                "in the feature's descriptors.",
            )
        return [str(s) for s in feature.descriptors[self.structure]]

    def prepare(self, structures: pd.Series) -> None:
        key = tuple(structures.tolist())
        if key not in self._filtered_cache:
            # reset any stale filter state, then compute for this structure set
            self.generator._descriptors = None
            self.generator.remove_correlated_descriptors(list(key))
            self._filtered_cache[key] = self.generator._descriptors
        # point the generator at this set's descriptors before any use
        self.generator._descriptors = self._filtered_cache[key]

    def names(self, feature: "DescriptorsMixin") -> List[str]:
        self.prepare(pd.Series(self._structure_column(feature)))
        return list(self.generator.get_descriptor_names())

    def declared_names(self) -> Optional[List[str]]:
        # reflects the generator's current (possibly filtered) descriptor set.
        return list(self.generator.get_descriptor_names())

    def check(self, feature: "DescriptorsMixin") -> None:
        self._structure_column(feature)

    def table(self, feature: "DescriptorsMixin") -> pd.DataFrame:
        structures = self._structure_column(feature)
        self.prepare(pd.Series(structures))
        df = self.generator.get_descriptor_values(pd.Series(structures))
        df.index = feature.descriptor_levels()
        df.columns = self.generator.get_descriptor_names()
        return df

    def component_table(self, features: Sequence["DescriptorsMixin"]) -> pd.DataFrame:
        # one structure per component; filter once over the combined set so every
        # component row shares the same (filtered) descriptor columns.
        structures = [self._structure_column(feature)[0] for feature in features]
        self.prepare(pd.Series(structures))
        df = self.generator.get_descriptor_values(pd.Series(structures))
        df.index = [feature.key for feature in features]
        df.columns = self.generator.get_descriptor_names()
        return df
