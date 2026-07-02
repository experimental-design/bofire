"""Base class for categorical encodings.

An encoding is the *per-surrogate choice* of how a categorical feature's value is
turned into model-input columns. It reads the *data* carried on the feature (its
``categories`` and ``descriptors`` table) and holds the modelling knobs. The
mechanical pandas transforms themselves live on ``CategoricalInput`` (so they
stay available downstream); the encoders are thin choice wrappers that delegate.
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
    def get_names(self, feature: "CategoricalInput") -> List[str]:
        """Final (encoded) column names produced for ``feature``."""

    @abstractmethod
    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        """Encode a series of categories into model-input columns.

        The returned frame's columns equal ``get_names(feature)``.
        """

    @abstractmethod
    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        """Back-transform encoded columns to categories."""

    @abstractmethod
    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        """Lower/upper bounds of the encoded columns for ``feature``.

        If ``values`` is None the optimization bounds (over allowed categories)
        are returned, else the full bounds over all categories.
        """

    def validate_for_feature(self, feature: "CategoricalInput") -> None:
        """Validate this encoder instance is applicable to ``feature``.

        The type-level check (``valid_transform_types``) only ensures the encoder
        *class* is allowed; this instance-level hook rejects encoders whose concrete
        configuration cannot produce columns for the feature (e.g. a static
        descriptor source on a feature that only carries a structure column).
        """
