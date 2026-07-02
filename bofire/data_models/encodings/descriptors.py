"""Descriptor encoding: encode a category via its descriptor row.

The descriptor columns are declared by the :class:`DescriptorSpec` mixin (static
columns and/or molecular generators); the encoding itself (select-row on encode,
nearest-neighbour on decode, min/max bounds) is the same however they are produced.
"""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding
from bofire.data_models.encodings.naming import get_encoded_name
from bofire.data_models.features._descriptor_spec import DescriptorSpec


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class DescriptorEncoding(CategoricalEncoding, DescriptorSpec):
    """Encode categories via a per-category descriptor row.

    Inherits ``columns`` / ``generators`` / ``filter_descriptors`` /
    ``correlation_cutoff`` from :class:`DescriptorSpec`. With the default (no static
    columns listed, no generators) it uses all of the feature's numeric descriptor
    columns.
    """

    type: Literal["DescriptorEncoding"] = "DescriptorEncoding"

    def validate_for_feature(self, feature: "CategoricalInput") -> None:
        self.validate_for(feature)

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        # avoid running generators when not filtering (declared names suffice).
        if self.filter_descriptors:
            names = list(self.table(feature).columns)
        else:
            names = self.declared_names(feature) or []
        return [get_encoded_name(feature.key, d) for d in names]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        table = self.table(feature)
        return pd.DataFrame(
            data=table.loc[values.tolist()].to_numpy(),
            columns=[get_encoded_name(feature.key, d) for d in table.columns],
            index=values.index,
        )

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        table = self.table(feature)
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
        table_allowed = self.table(feature).loc[allowed]
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
