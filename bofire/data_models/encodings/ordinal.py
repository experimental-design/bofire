"""Ordinal (integer-index) encoding — the pre-model identity encoding."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class OrdinalEncoding(CategoricalEncoding):
    """Encode a category as its integer index ``0 .. len(categories) - 1``."""

    type: Literal["OrdinalEncoding"] = "OrdinalEncoding"

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        return [feature.key]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        enc = pd.Series(range(len(feature.categories)), index=list(feature.categories))
        s = enc[values]
        s.index = values.index
        s.name = feature.key
        return s.to_frame()

    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        codes = values[feature.key].astype(int)
        return pd.Series(
            np.array(feature.categories)[codes],
            index=values.index,
            name=feature.key,
        )

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        return [0.0], [float(len(feature.categories) - 1)]
