"""Base class for categorical encodings.

An encoding is the *per-surrogate choice* of how a categorical feature's value is
turned into model-input columns. It reads the *data* carried on the feature (its
``descriptors`` table) and holds the modelling knobs (which columns, which
generator, hyperparameters). This inverts the previous design where the feature
class itself knew how to encode.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import pandas as pd

from bofire.data_models.base import BaseModel


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class CategoricalEncoding(BaseModel):
    """Base class for all categorical encodings."""

    type: Any

    @abstractmethod
    def get_descriptor_names(self, feature: "CategoricalInput") -> List[str]:
        """Unprefixed names of the columns produced for ``feature``."""

    @abstractmethod
    def to_descriptor_encoding(
        self,
        feature: "CategoricalInput",
        values: pd.Series,
    ) -> pd.DataFrame:
        """Encode a series of categories into descriptor columns.

        The returned frame uses ``get_encoded_name(feature.key, name)`` columns.
        """

    @abstractmethod
    def from_descriptor_encoding(
        self,
        feature: "CategoricalInput",
        values: pd.DataFrame,
    ) -> pd.Series:
        """Back-transform descriptor columns to the nearest categories."""

    @abstractmethod
    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        """Descriptor-space lower/upper bounds for ``feature``.

        If ``values`` is None the optimization bounds (over allowed categories)
        are returned, else the full bounds over all categories.
        """
