import warnings
from typing import ClassVar, Literal

import pandas as pd
from pydantic import model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput


class ContinuousDescriptorInput(ContinuousInput):
    """Deprecated. Use :class:`ContinuousInput` with a ``descriptors`` table instead.

    Kept as a thin deserialization shim: the legacy ``descriptors`` (list of names)
    + ``values`` (single row) input is rewritten into the base ``descriptors`` dict
    (single-element columns, since a continuous feature is one component).
    """

    type: Literal["ContinuousDescriptorInput"] = "ContinuousDescriptorInput"
    order_id: ClassVar[int] = 2

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_descriptors(cls, data):
        if not isinstance(data, dict):
            return data
        warnings.warn(
            "`ContinuousDescriptorInput` is deprecated, use `ContinuousInput` "
            "with a `descriptors` table instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # legacy shape: descriptors=[names], values=[row]
        if "values" in data or isinstance(data.get("descriptors"), list):
            names = data.pop("descriptors")
            values = data.pop("values")
            data["descriptors"] = {name: [values[j]] for j, name in enumerate(names)}
        return data


class CategoricalDescriptorInput(CategoricalInput):
    """Deprecated. Use :class:`CategoricalInput` with a ``descriptors`` table instead.

    Kept as a thin deserialization shim: the legacy ``descriptors`` (list of names)
    + ``values`` (rows per category) input is rewritten into the base
    ``descriptors`` dict and re-emitted in the new shape.
    """

    type: Literal["CategoricalDescriptorInput"] = "CategoricalDescriptorInput"
    order_id: ClassVar[int] = 6

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_descriptors(cls, data):
        if not isinstance(data, dict):
            return data
        warnings.warn(
            "`CategoricalDescriptorInput` is deprecated, use `CategoricalInput` "
            "with a `descriptors` table instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # legacy shape: descriptors=[names], values=[[row per category]]
        if "values" in data or isinstance(data.get("descriptors"), list):
            names = data.pop("descriptors")
            values = data.pop("values")
            data["descriptors"] = {
                name: [row[j] for row in values] for j, name in enumerate(names)
            }
        return data

    @classmethod
    def from_df(cls, key: str, df: pd.DataFrame):
        """Creates a feature from a dataframe with categories as rows and
        descriptors as columns.
        """
        return cls(
            key=key,
            categories=list(df.index),
            allowed=[True for _ in range(len(df))],
            descriptors={col: df[col].tolist() for col in df.columns},
        )
