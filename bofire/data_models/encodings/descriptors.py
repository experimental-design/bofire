"""Descriptor encoding: encode a category by looking up its numeric descriptor row."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class DescriptorEncoding(CategoricalEncoding):
    """Encode categories via (a subset of) the feature's numeric descriptor columns.

    Attributes:
        columns: Names of the numeric descriptor columns to use. ``None`` means
            all non-reserved (role ``"descriptor"``) columns in declared order.
    """

    type: Literal["DescriptorEncoding"] = "DescriptorEncoding"
    columns: Optional[List[str]] = None

    def _columns(self, feature: "CategoricalInput") -> List[str]:
        if self.columns is None:
            return feature.descriptor_columns(role="descriptor")
        available = set(feature.descriptor_columns(role="descriptor"))
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise ValueError(
                f"{feature.key}: descriptor columns {missing} are not available "
                f"as numeric descriptors. Available: {sorted(available)}.",
            )
        return list(self.columns)

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        from bofire.data_models.features.feature import get_encoded_name

        return [get_encoded_name(feature.key, d) for d in self._columns(feature)]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        table = feature.descriptor_table(self._columns(feature))
        return pd.DataFrame(
            data=table.loc[values.tolist()].to_numpy(),
            columns=self.get_names(feature),
            index=values.index,
        )

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        table = feature.descriptor_table(self._columns(feature))
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
        table_allowed = feature.descriptor_table(self._columns(feature)).loc[allowed]
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
