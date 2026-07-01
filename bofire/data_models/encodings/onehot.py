"""One-hot (and dummy / drop-first) encoding."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding


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

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        from bofire.data_models.features.feature import get_encoded_name

        categories = feature.categories[1:] if self.drop_first else feature.categories
        return [get_encoded_name(feature.key, c) for c in categories]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        if self.drop_first:
            return feature.to_dummy_encoding(values)
        return feature.to_onehot_encoding(values)

    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        if self.drop_first:
            return feature.from_dummy_encoding(values)
        return feature.from_onehot_encoding(values)

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
