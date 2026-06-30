import warnings
from typing import ClassVar, List, Literal

import pandas as pd
from pydantic import model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.types import Descriptors, DiscreteVals


class ContinuousDescriptorInput(ContinuousInput):
    """Class for continuous input features with descriptors

    Attributes:
        lower_bound (float): Lower bound of the feature in the optimization.
        upper_bound (float): Upper bound of the feature in the optimization.
        descriptors (List[str]): Names of the descriptors.
        values (List[float]): Values of the descriptors.

    """

    type: Literal["ContinuousDescriptorInput"] = "ContinuousDescriptorInput"
    order_id: ClassVar[int] = 2

    descriptors: Descriptors
    values: DiscreteVals

    def _extra_description_parts(self) -> List[str]:
        return [f"descriptors: {dict(zip(self.descriptors, self.values))}"]

    @model_validator(mode="after")
    def validate_list_lengths(self):
        """Compares the length of the defined descriptors list with the provided values

        Args:
            values (Dict): Dictionary with all attributes

        Raises:
            ValueError: when the number of descriptors does not math the number of provided values

        Returns:
            Dict: Dict with the attributes

        """
        if len(self.descriptors) != len(self.values):
            raise ValueError(
                'must provide same number of descriptors and values, got {len(values["descriptors"])} != {len(values["values"])}',
            )
        return self

    def to_df(self) -> pd.DataFrame:
        """Tabular overview of the feature as DataFrame

        Returns:
            pd.DataFrame: tabular overview of the feature as DataFrame

        """
        return pd.DataFrame(
            data=[self.values],
            index=[self.key],
            columns=self.descriptors,
        )


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
