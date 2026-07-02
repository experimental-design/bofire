"""One-hot (and dummy / drop-first) encoding."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding
from bofire.data_models.encodings.naming import get_encoded_name


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class OneHotEncoding(CategoricalEncoding):
    """One-hot encode a category into one binary column per category.

    Attributes:
        drop_first: If True, drop the first category's column (dummy encoding),
            yielding ``len(categories) - 1`` columns.
    """

    type: Literal["OneHotEncoding"] = "OneHotEncoding"
    drop_first: bool = False

    def _categories(self, feature: "CategoricalInput") -> list:
        return feature.categories[1:] if self.drop_first else feature.categories

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        return [get_encoded_name(feature.key, c) for c in self._categories(feature)]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                name: (values == category)
                for name, category in zip(
                    self.get_names(feature), self._categories(feature)
                )
            },
            dtype=float,
            index=values.index,
        )

    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        cat_cols = [get_encoded_name(feature.key, c) for c in feature.categories]
        # for the dropped-first column we only require the remaining columns; it is
        # reconstructed below. we allow more columns than needed to ease back-transform.
        required = cat_cols[1:] if self.drop_first else cat_cols
        if np.any([c not in values.columns for c in required]):
            raise ValueError(
                f"{feature.key}: Column names don't match categorical levels: "
                f"{values.columns}, {required}.",
            )
        if self.drop_first:
            values = values.copy()
            values[cat_cols[0]] = 1 - values[cat_cols[1:]].sum(axis=1)
        s = values[cat_cols].idxmax(1).str[(len(feature.key) + 1) :]
        s.name = feature.key
        return s

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        if self.drop_first:
            n = len(feature.categories) - 1
            return [0.0] * n, [1.0] * n
        lower = [0.0 for _ in feature.categories]
        if values is None:
            # optimization bounds: forbidden categories are pinned to 0.
            upper = [
                1.0 if feature.allowed[i] else 0.0  # ty: ignore[not-subscriptable]
                for i, _ in enumerate(feature.categories)
            ]
        else:
            upper = [1.0 for _ in feature.categories]
        return lower, upper
