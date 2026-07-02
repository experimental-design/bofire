"""Descriptor encoding: encode a category via its descriptor row from a source."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.descriptors.api import AnyDescriptorSource
from bofire.data_models.descriptors.static import StaticSource
from bofire.data_models.encodings.encoding import CategoricalEncoding
from bofire.data_models.encodings.naming import get_encoded_name


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class DescriptorEncoding(CategoricalEncoding):
    """Encode categories via a per-category descriptor row from a descriptor source.

    The ``source`` (static columns / molecular generator / composite) decides how
    the descriptor columns are produced; the encoding itself (select-row on encode,
    nearest-neighbour on decode, min/max bounds) is the same for every source.
    """

    type: Literal["DescriptorEncoding"] = "DescriptorEncoding"
    # default: use the feature's own static descriptor columns (legacy zero-arg shape).
    source: AnyDescriptorSource = Field(default_factory=StaticSource)

    def validate_for_feature(self, feature: "CategoricalInput") -> None:
        self.source.check(feature)

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        return [get_encoded_name(feature.key, d) for d in self.source.names(feature)]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        table = self.source.table(feature)
        out = pd.DataFrame(
            data=table.loc[values.tolist()].to_numpy(),
            columns=self.get_names(feature),
            index=values.index,
        )
        return out

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        table = self.source.table(feature)
        # values None -> optimization bounds over allowed categories,
        # else full bounds over all categories (for model fitting).
        if values is None:
            table = table.loc[feature.get_allowed_categories()]
        return table.min().values.tolist(), table.max().values.tolist()

    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        cat_cols = self.get_names(feature)
        # we allow here explicitly that the dataframe can have more columns than
        # needed to make the back-transform easier.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{feature.key}: Column names don't match categorical levels: "
                f"{values.columns}, {cat_cols}.",
            )
        allowed = feature.get_allowed_categories()
        table_allowed = self.source.table(feature).loc[allowed]
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - table_allowed.to_numpy()
                    )
                    ** 2,
                    axis=2,
                ),
            ),
            columns=allowed,
            index=values.index,
        ).idxmin(1)
        s.name = feature.key
        return s
