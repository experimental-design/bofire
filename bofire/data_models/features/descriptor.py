from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator, validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.feature import (
    _CAT_SEP,
    TCategoricalDescriptorVals,
    TDescriptors,
    TDiscreteVals,
    TTransform,
)


# TODO: write a Descriptor base class from which both Categorical and Continuous Descriptor are inheriting
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

    descriptors: TDescriptors
    values: TDiscreteVals

    @validator("descriptors")
    def descriptors_to_keys(cls, descriptors):
        """validates the descriptor names and transforms it to valid keys

        Args:
            descriptors (List[str]): List of descriptor names

        Returns:
            List[str]: List of valid keys
        """
        return list(descriptors)

    @root_validator(pre=False, skip_on_failure=True)
    def validate_list_lengths(cls, values):
        """compares the length of the defined descriptors list with the provided values

        Args:
            values (Dict): Dictionary with all attribues

        Raises:
            ValueError: when the number of descriptors does not math the number of provided values

        Returns:
            Dict: Dict with the attributes
        """
        if len(values["descriptors"]) != len(values["values"]):
            raise ValueError(
                'must provide same number of descriptors and values, got {len(values["descriptors"])} != {len(values["values"])}'
            )
        return values

    def to_df(self) -> pd.DataFrame:
        """tabular overview of the feature as DataFrame

        Returns:
            pd.DataFrame: tabular overview of the feature as DataFrame
        """
        return pd.DataFrame(
            data=[self.values], index=[self.key], columns=self.descriptors
        )


class CategoricalDescriptorInput(CategoricalInput):
    """Class for categorical input features with descriptors

    Attributes:
        categories (List[str]): Names of the categories.
        allowed (List[bool]): List of bools indicating if a category is allowed within the optimization.
        descriptors (List[str]): List of strings representing the names of the descriptors.
        values (List[List[float]]): List of lists representing the descriptor values.
    """

    type: Literal["CategoricalDescriptorInput"] = "CategoricalDescriptorInput"
    order_id: ClassVar[int] = 4

    descriptors: TDescriptors
    values: TCategoricalDescriptorVals

    @validator("descriptors")
    def validate_descriptors(cls, descriptors):
        """validates that descriptors have unique names

        Args:
            categories (List[str]): List of descriptor names

        Raises:
            ValueError: when descriptors have non-unique names

        Returns:
            List[str]: List of the descriptors
        """
        descriptors = list(descriptors)
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("descriptors must be unique")
        return descriptors

    @validator("values")
    def validate_values(cls, v, values):
        """validates the compatability of passed values for the descriptors and the defined categories

        Args:
            v (List[List[float]]): Nested list with descriptor values
            values (Dict): Dictionary with attributes

        Raises:
            ValueError: when values have different length than categories
            ValueError: when rows in values have different length than descriptors
            ValueError: when a descriptor shows no variance in the data

        Returns:
            List[List[float]]: Nested list with descriptor values
        """
        if len(v) != len(values["categories"]):
            raise ValueError("values must have same length as categories")
        for row in v:
            if len(row) != len(values["descriptors"]):
                raise ValueError("rows in values must have same length as descriptors")
        a = np.array(v)
        for i, d in enumerate(values["descriptors"]):
            if len(set(a[:, i])) == 1:
                raise ValueError(f"No variation for descriptor {d}.")
        return v

    @staticmethod
    def valid_transform_types() -> List[CategoricalEncodingEnum]:
        return [
            CategoricalEncodingEnum.ONE_HOT,
            CategoricalEncodingEnum.DUMMY,
            CategoricalEncodingEnum.ORDINAL,
            CategoricalEncodingEnum.DESCRIPTOR,
        ]

    def to_df(self):
        """tabular overview of the feature as DataFrame

        Returns:
            pd.DataFrame: tabular overview of the feature as DataFrame
        """
        data = dict(zip(self.categories, self.values))
        return pd.DataFrame.from_dict(data, orient="index", columns=self.descriptors)

    def fixed_value(
        self, transform_type: Optional[TTransform] = None
    ) -> Union[List[str], List[float], None]:
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None
        """
        if transform_type != CategoricalEncodingEnum.DESCRIPTOR:
            return super().fixed_value(transform_type)
        else:
            val = self.get_allowed_categories()[0]
            return self.to_descriptor_encoding(pd.Series([val])).values[0].tolist()

    def get_bounds(
        self, transform_type: TTransform, values: Optional[pd.Series] = None
    ) -> Tuple[List[float], List[float]]:
        if transform_type != CategoricalEncodingEnum.DESCRIPTOR:
            return super().get_bounds(transform_type, values)
        else:
            # in case that values is None, we return the optimization bounds
            # else we return the complete bounds
            if values is None:
                df = self.to_df().loc[self.get_allowed_categories()]
            else:
                df = self.to_df()
            lower = df.min().values.tolist()  # type: ignore
            upper = df.max().values.tolist()  # type: ignore
            return lower, upper

    def validate_experimental(
        self, values: pd.Series, strict: bool = False
    ) -> pd.Series:
        """Method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurence of fixed features in the dataset should be considered or not. Defaults to False.

        Raises:
            ValueError: when an entry is not in the list of allowed categories
            ValueError: when there is no variation in a feature provided by the experimental data
            ValueError: when no variation is present or planed for a given descriptor

        Returns:
            pd.Series: A dataFrame with experiments
        """
        values = super().validate_experimental(values, strict)
        if strict:
            lower, upper = self.get_bounds(
                transform_type=CategoricalEncodingEnum.DESCRIPTOR, values=values
            )
            for i, desc in enumerate(self.descriptors):
                if lower[i] == upper[i]:
                    raise ValueError(
                        f"No variation present or planned for descriptor {desc} for feature {self.key}. Remove the descriptor."
                    )
        return values

    @classmethod
    def from_df(cls, key: str, df: pd.DataFrame):
        """Creates a feature from a dataframe

        Args:
            key (str): The name of the feature
            df (pd.DataFrame): Categories as rows and descriptors as columns

        Returns:
            _type_: _description_
        """
        return cls(
            key=key,
            categories=list(df.index),
            allowed=[True for _ in range(len(df))],
            descriptors=list(df.columns),
            values=df.values.tolist(),
        )

    def to_descriptor_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to descriptor encoding.

        Args:
            values (pd.Series): Values to transform.

        Returns:
            pd.DataFrame: Descriptor encoded dataframe.
        """
        return pd.DataFrame(
            data=values.map(
                dict(zip(self.categories, self.values))
            ).values.tolist(),  # type: ignore
            columns=[f"{self.key}{_CAT_SEP}{d}" for d in self.descriptors],
            index=values.index,
        )

    def from_descriptor_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Converts values back from descriptor encoding.

        Args:
            values (pd.DataFrame): Descriptor encoded dataframe.

        Raises:
            ValueError: If descriptor columns not found in the dataframe.

        Returns:
            pd.Series: Series with categorical values.
        """
        cat_cols = [f"{self.key}{_CAT_SEP}{d}" for d in self.descriptors]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}."
            )
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - self.to_df().iloc[self.allowed].to_numpy()
                    )
                    ** 2,
                    axis=2,
                )
            ),
            columns=self.get_allowed_categories(),
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s
