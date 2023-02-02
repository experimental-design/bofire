from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field, parse_obj_as, validate_arguments, validator
from pydantic.class_validators import root_validator
from pydantic.types import conint, conlist
from scipy.stats.qmc import LatinHypercube, Sobol

from bofire.domain.objectives import (
    AnyAbstractObjective,
    AnyObjective,
    MaximizeObjective,
    Objective,
)
from bofire.domain.util import (
    KeyModel,
    PydanticBaseModel,
    filter_by_attribute,
    filter_by_class,
    is_numeric,
    name2key,
)
from bofire.utils.enum import CategoricalEncodingEnum, SamplingMethodEnum, ScalerEnum

_CAT_SEP = "_"


TTransform = Union[CategoricalEncodingEnum, ScalerEnum]
TInputTransformSpecs = Dict[str, CategoricalEncodingEnum]


class Feature(KeyModel):
    """The base class for all features."""

    type: str

    def __lt__(self, other) -> bool:
        """
        Method to compare two models to get them in the desired order.
        Return True if other is larger than self, else False. (see FEATURE_ORDER)

        Args:
            other: The other class to compare to self

        Returns:
            bool: True if the other class is larger than self, else False
        """
        # TODO: add order of base class to FEATURE_ORDER and remove type: ignore
        order_self = FEATURE_ORDER[type(self)]  # type: ignore
        order_other = FEATURE_ORDER[type(other)]
        if order_self == order_other:
            return self.key < other.key
        else:
            return order_self < order_other

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(AnyFeature, dict_)


class InputFeature(Feature):
    """Base class for all input features."""

    type: Literal["InputFeature"] = "InputFeature"

    @abstractmethod
    def is_fixed(self) -> bool:
        """Indicates if a variable is set to a fixed value.

        Returns:
            bool: True if fixed, els False.
        """
        pass

    @abstractmethod
    def fixed_value(
        self, transform_type: Optional[TTransform] = None
    ) -> Union[None, List[str], List[float]]:
        """Method to return the fixed value in case of a fixed feature.

        Returns:
            Union[None,str,float]: None in case the feature is not fixed, else the fixed value.
        """
        pass

    @abstractmethod
    def validate_experimental(
        self, values: pd.Series, strict: bool = False
    ) -> pd.Series:
        """Abstract method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurence of fixed features in the dataset should be considered or not. Defaults to False.

        Returns:
            pd.Series: The passed dataFrame with experiments
        """
        pass

    @abstractmethod
    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Abstract method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Returns:
            pd.Series: The passed dataFrame with candidates
        """
        pass

    @abstractmethod
    def sample(self, n: int) -> pd.Series:
        """Sample a series of allowed values.

        Args:
            n (int): Number of samples

        Returns:
            pd.Series: Sampled values.
        """
        pass

    @abstractmethod
    def get_bounds(
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        """Returns the bounds of an input feature depending on the requested transform type.

        Args:
            transform_type (Optional[TTransform], optional): The requested transform type. Defaults to None.
            values (Optional[pd.Series], optional): If values are provided the bounds are returned taking
                the most extreme values for the feature into account. Defaults to None.

        Returns:
            Tuple[List[float], List[float]]: List of lower bound values, list of upper bound values.
        """
        pass


class NumericalInput(InputFeature):
    """Abstract base class for all numerical (ordinal) input features."""

    type: Literal["NumericalInput"] = "NumericalInput"

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(AnyInputFeature, dict_)

    def to_unit_range(
        self, values: Union[pd.Series, np.ndarray], use_real_bounds: bool = False
    ) -> Union[pd.Series, np.ndarray]:
        """Convert to the unit range between 0 and 1.

        Args:
            values (pd.Series): values to be transformed
            use_real_bounds (bool, optional): if True, use the bounds from the actual values else the bounds from the feature.
                Defaults to False.

        Raises:
            ValueError: If lower_bound == upper bound an error is raised

        Returns:
            pd.Series: transformed values.
        """
        if use_real_bounds:
            lower, upper = self.get_bounds(transform_type=None, values=values)
            lower = lower[0]
            upper = upper[0]
        else:
            lower, upper = self.lower_bound, self.upper_bound  # type: ignore
        if lower == upper:
            raise ValueError("Fixed feature cannot be transformed to unit range.")
        valrange = upper - lower
        return (values - lower) / valrange

    def from_unit_range(
        self, values: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """Convert from unit range.

        Args:
            values (pd.Series): values to transform from.

        Raises:
            ValueError: if the feature is fixed raise a value error.

        Returns:
            pd.Series: _description_
        """
        if self.is_fixed():
            raise ValueError("Fixed feature cannot be transformed from unit range.")
        valrange = self.upper_bound - self.lower_bound  # type: ignore
        return (values * valrange) + self.lower_bound  # type: ignore

    def is_fixed(self):
        """Method to check if the feature is fixed

        Returns:
            Boolean: True when the feature is fixed, false otherwise.
        """
        # TODO: the bounds are declared in the derived classes, hence the type checks fail here :(.
        return self.lower_bound == self.upper_bound  # type: ignore

    def fixed_value(
        self, transform_type: Optional[TTransform] = None
    ) -> Union[None, List[float]]:
        """Method to get the value to which the feature is fixed

        Returns:
            Float: Return the feature value or None if the feature is not fixed.
        """
        assert transform_type is None
        if self.is_fixed():
            return [self.lower_bound]  # type: ignore
        else:
            return None

    def validate_experimental(self, values: pd.Series, strict=False) -> pd.Series:
        """Method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurence of fixed features in the dataset should be considered or not.
                Defaults to False.

        Raises:
            ValueError: when a value is not numerical
            ValueError: when there is no variation in a feature provided by the experimental data

        Returns:
            pd.Series: A dataFrame with experiments
        """
        if not is_numeric(values):
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical"
            )
        if strict:
            lower, upper = self.get_bounds(transform_type=None, values=values)
            if lower == upper:
                raise ValueError(
                    f"No variation present or planned for feature {self.key}. Remove it."
                )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Validate the suggested candidates for the feature.

        Args:
            values (pd.Series): suggested candidates for the feature

        Raises:
            ValueError: Error is raised when one of the values is not numerical.

        Returns:
            pd.Series: the original provided candidates
        """
        if not is_numeric(values):
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical"
            )
        return values

    def get_bounds(
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        assert transform_type is None
        if values is None:
            return [self.lower_bound], [self.upper_bound]  # type: ignore
        lower = min(self.lower_bound, values.min())  # type: ignore
        upper = max(self.upper_bound, values.max())  # type: ignore
        return [lower], [upper]  # type: ignore


class ContinuousInput(NumericalInput):
    """Base class for all continuous input features.

    Attributes:
        lower_bound (float): Lower bound of the feature in the optimization.
        upper_bound (float): Upper bound of the feature in the optimization.
    """

    type: Literal["ContinuousInput"] = "ContinuousInput"
    lower_bound: float
    upper_bound: float

    @root_validator(pre=False, skip_on_failure=True)
    def validate_lower_upper(cls, values):
        """Validates that the lower bound is lower than the upper bound

        Args:
            values (Dict): Dictionary with attributes key, lower and upper bound

        Raises:
            ValueError: when the lower bound is higher than the upper bound

        Returns:
            Dict: The attributes as dictionary
        """
        if values["lower_bound"] > values["upper_bound"]:
            raise ValueError(
                f'lower bound must be <= upper bound, got {values["lower_bound"]} > {values["upper_bound"]}'
            )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when non numerical values are passed
            ValueError: when values are larger than the upper bound of the feature
            ValueError: when values are lower than the lower bound of the feature

        Returns:
            pd.Series: The passed dataFrame with candidates
        """
        noise = 10e-6
        super().validate_candidental(values)
        if (values < self.lower_bound - noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are larger than lower bound `{self.lower_bound}` "
            )
        if (values > self.upper_bound + noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are smaller than upper bound `{self.upper_bound}` "
            )
        return values

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key,
            data=np.random.uniform(self.lower_bound, self.upper_bound, n),
        )

    def __str__(self) -> str:
        """Method to return a string of lower and upper bound

        Returns:
            str: String of a list with lower and upper bound
        """
        return f"[{self.lower_bound},{self.upper_bound}]"


TDiscreteVals = conlist(item_type=float, min_items=1)


class DiscreteInput(NumericalInput):
    """Feature with discretized ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the discretized allowed values during the optimization.
    """

    type: Literal["DiscreteInput"] = "DiscreteInput"
    values: TDiscreteVals

    @validator("values")
    def validate_values_unique(cls, values):
        """Validates that provided values are unique.

        Args:
            values (List[float]): List of values

        Raises:
            ValueError: when values are non-unique.

        Returns:
            List[values]: Sorted list of values
        """
        if len(values) != len(set(values)):
            raise ValueError("Discrete values must be unique")
        return sorted(values)

    @property
    def lower_bound(self) -> float:
        """Lower bound of the set of allowed values"""
        return min(self.values)

    @property
    def upper_bound(self) -> float:
        """Upper bound of the set of allowed values"""
        return max(self.values)

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the provided candidates.

        Args:
            values (pd.Series): suggested candidates for the feature

        Raises:
            ValueError: Raises error when one of the provided values is not contained in the list of allowed values.

        Returns:
            pd.Series: _uggested candidates for the feature
        """
        super().validate_candidental(values)
        if not np.isin(values.to_numpy(), np.array(self.values)).all():
            raise ValueError(
                f"Not allowed values in candidates for feature {self.key}."
            )
        return values

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(name=self.key, data=np.random.choice(self.values, n))


TDescriptors = conlist(item_type=str, min_items=1)


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
        return [name2key(name) for name in descriptors]

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


TCategoryVals = conlist(item_type=str, min_items=2)
TAllowedVals = Optional[conlist(item_type=bool, min_items=2)]


class CategoricalInput(InputFeature):
    """Base class for all categorical input features.

    Attributes:
        categories (List[str]): Names of the categories.
        allowed (List[bool]): List of bools indicating if a category is allowed within the optimization.
    """

    type: Literal["CategoricalInput"] = "CategoricalInput"
    categories: TCategoryVals
    allowed: TAllowedVals = None

    @validator("categories")
    def validate_categories_unique(cls, categories):
        """validates that categories have unique names

        Args:
            categories (List[str]): List of category names

        Raises:
            ValueError: when categories have non-unique names

        Returns:
            List[str]: List of the categories
        """
        categories = [name2key(name) for name in categories]
        if len(categories) != len(set(categories)):
            raise ValueError("categories must be unique")
        return categories

    @root_validator(pre=False, skip_on_failure=True)
    def init_allowed(cls, values):
        """validates the list of allowed/not allowed categories

        Args:
            values (Dict): Dictionary with attributes

        Raises:
            ValueError: when the number of allowences does not fit to the number of categories
            ValueError: when no category is allowed

        Returns:
            Dict: Dictionary with attributes
        """
        if "categories" not in values or values["categories"] is None:
            return values
        if "allowed" not in values or values["allowed"] is None:
            values["allowed"] = [True for _ in range(len(values["categories"]))]
        if len(values["allowed"]) != len(values["categories"]):
            raise ValueError("allowed must have same length as categories")
        if sum(values["allowed"]) == 0:
            raise ValueError("no category is allowed")
        return values

    def is_fixed(self) -> bool:
        """Returns True if there is only one allowed category.

        Returns:
            [bool]: True if there is only one allowed category
        """
        if self.allowed is None:
            return False
        return sum(self.allowed) == 1

    def fixed_value(
        self, transform_type: Optional[TTransform] = None
    ) -> Union[List[str], List[float], None]:
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None
        """
        if self.is_fixed():
            val = self.get_allowed_categories()[0]
            if transform_type is None:
                return [val]
            elif transform_type == CategoricalEncodingEnum.ONE_HOT:
                return self.to_onehot_encoding(pd.Series([val])).values[0].tolist()
            elif transform_type == CategoricalEncodingEnum.DUMMY:
                return self.to_dummy_encoding(pd.Series([val])).values[0].tolist()
            elif transform_type == CategoricalEncodingEnum.ORDINAL:
                return self.to_ordinal_encoding(pd.Series([val])).tolist()
            else:
                raise ValueError(
                    f"Unkwon transform type {transform_type} for categorical input {self.key}"
                )
        else:
            return None

    def get_allowed_categories(self):
        """Returns the allowed categories.

        Returns:
            list of str: The allowed categories
        """
        if self.allowed is None:
            return []
        return [c for c, a in zip(self.categories, self.allowed) if a]

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

        Returns:
            pd.Series: A dataFrame with experiments
        """
        if sum(values.isin(self.categories)) != len(values):
            raise ValueError(
                f"invalid values for `{self.key}`, allowed are: `{self.categories}`"
            )
        if strict:
            possible_categories = self.get_possible_categories(values)
            if len(possible_categories) != len(self.categories):
                raise ValueError(
                    f"Categories {list(set(self.categories)-set(possible_categories))} of feature {self.key} not used. Remove them."
                )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when not all values for a feature are one of the allowed categories

        Returns:
            pd.Series: The passed dataFrame with candidates
        """
        if sum(values.isin(self.get_allowed_categories())) != len(values):
            raise ValueError(
                f"not all values of input feature `{self.key}` are a valid allowed category from {self.get_allowed_categories()}"
            )
        return values

    def get_forbidden_categories(self):
        """Returns the non-allowed categories

        Returns:
            List[str]: List of the non-allowed categories
        """
        return list(set(self.categories) - set(self.get_allowed_categories()))

    def get_possible_categories(self, values: pd.Series) -> list:
        """Return the superset of categories that have been used in the experimental dataset and
        that can be used in the optimization

        Args:
            values (pd.Series): Series with the values for this feature

        Returns:
            list: list of possible categories
        """
        return sorted(
            list(set(list(set(values.tolist())) + self.get_allowed_categories()))
        )

    def to_onehot_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a one-hot encoding.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: One-hot transformed data frame.
        """
        return pd.DataFrame(
            {f"{self.key}{_CAT_SEP}{c}": values == c for c in self.categories},
            dtype=float,
            index=values.index,
        )

    def from_onehot_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Converts values back from one-hot encoding.

        Args:
            values (pd.DataFrame): One-hot encoded values.

        Raises:
            ValueError: If one-hot columns not present in `values`.

        Returns:
            pd.Series: Series with categorical values.
        """
        cat_cols = [f"{self.key}{_CAT_SEP}{c}" for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}."
            )
        return pd.Series(
            [i.split(_CAT_SEP)[-1] for i in values[cat_cols].idxmax(1).to_list()],
            name=self.key,
            index=values.index,
        )

    def to_dummy_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a dummy-hot encoding, dropping the first categorical level.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: Dummy-hot transformed data frame.
        """
        return pd.DataFrame(
            {f"{self.key}{_CAT_SEP}{c}": values == c for c in self.categories[1:]},
            dtype=float,
            index=values.index,
        )

    def from_dummy_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Convert points back from dummy encoding.

        Args:
            values (pd.DataFrame): Dummy-hot encoded values.

        Raises:
            ValueError: If one-hot columns not present in `values`.

        Returns:
            pd.Series: Series with categorical values.
        """
        cat_cols = [f"{self.key}{_CAT_SEP}{c}" for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols[1:]]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols[1:]}."
            )
        values = values.copy()
        values[cat_cols[0]] = 1 - values[cat_cols[1:]].sum(axis=1)
        s = values[cat_cols].idxmax(1).str.split(_CAT_SEP, expand=True)[1]
        s.name = self.key
        return s

    def to_ordinal_encoding(self, values: pd.Series) -> pd.Series:
        """Converts values to an ordinal integer based encoding.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.Series: Ordinal encoded values.
        """
        enc = pd.Series(range(len(self.categories)), index=list(self.categories))
        s = enc[values]
        s.index = values.index
        s.name = self.key
        return s

    def from_ordinal_encoding(self, values: pd.Series) -> pd.Series:
        """Convertes values back from ordinal encoding.

        Args:
            values (pd.Series): Ordinal encoded series.

        Returns:
            pd.Series: Series with categorical values.
        """
        enc = np.array(self.categories)
        return pd.Series(enc[values], index=values.index, name=self.key)

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key, data=np.random.choice(self.get_allowed_categories(), n)
        )

    def get_bounds(
        self,
        transform_type: TTransform,
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        assert isinstance(transform_type, CategoricalEncodingEnum)
        if transform_type == CategoricalEncodingEnum.ORDINAL:
            return [0], [len(self.categories) - 1]
        if transform_type == CategoricalEncodingEnum.ONE_HOT:
            # in the case that values are None, we return the bounds
            # based on the optimization bounds, else we return the true
            # bounds as this is for model fitting.
            if values is None:
                lower = [0.0 for _ in self.categories]
                upper = [
                    1.0 if self.allowed[i] is True else 0.0  # type: ignore
                    for i, _ in enumerate(self.categories)
                ]
            else:
                lower = [0.0 for _ in self.categories]
                upper = [1.0 for _ in self.categories]
            return lower, upper
        if transform_type == CategoricalEncodingEnum.DUMMY:
            lower = [0.0 for _ in range(len(self.categories) - 1)]
            upper = [1.0 for _ in range(len(self.categories) - 1)]
            return lower, upper
        if transform_type == CategoricalEncodingEnum.DESCRIPTOR:
            raise ValueError(
                f"Invalid descriptor transform for categorical {self.key}."
            )
        else:
            raise ValueError(
                f"Invalid transform_type {transform_type} provided for categorical {self.key}."
            )

    def __str__(self) -> str:
        """Returns the number of categories as str

        Returns:
            str: Number of categories
        """
        return f"{len(self.categories)} categories"


TCategoricalDescriptorVals = conlist(
    item_type=conlist(item_type=float, min_items=1), min_items=1
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
        descriptors = [name2key(name) for name in descriptors]
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
                raise ValueError("No variation for descriptor {d}.")
        return v

    def to_df(self):
        """tabular overview of the feature as DataFrame

        Returns:
            pd.DataFrame: tabular overview of the feature as DataFrame
        """
        data = {cat: values for cat, values in zip(self.categories, self.values)}
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
                {cat: value for cat, value in zip(self.categories, self.values)}
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
                        - self.to_df().to_numpy()
                    )
                    ** 2,
                    axis=2,
                )
            ),
            columns=self.categories,
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s


class OutputFeature(Feature):
    """Base class for all output features.

    Attributes:
        key(str): Key of the Feature.
    """

    type: Literal["OutputFeature"] = "OutputFeature"
    objective: Optional[AnyObjective]

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(AnyOutputFeature, dict_)


class ContinuousOutput(OutputFeature):
    """The base class for a continuous output feature

    Attributes:
        objective (objective, optional): objective of the feature indicating in which direction it should be optimzed. Defaults to `MaximizeObjective`.
    """

    type: Literal["ContinuousOutput"] = "ContinuousOutput"
    objective: Optional[AnyObjective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0)
    )

    def plot(
        self,
        lower: float,
        upper: float,
        experiments: Optional[pd.DataFrame] = None,
        plot_details: bool = True,
        line_options: Optional[Dict] = None,
        scatter_options: Optional[Dict] = None,
        label_options: Optional[Dict] = None,
        title_options: Optional[Dict] = None,
    ):
        """Plot the assigned objective.

        Args:
            lower (float): lower bound for the plot
            upper (float): upper bound for the plot
            experiments (Optional[pd.DataFrame], optional): If provided, scatter also the historical data in the plot. Defaults to None.
        """
        if self.objective is None:
            raise ValueError(
                f"No objective assigned for ContinuousOutputFeauture with key {self.key}."
            )

        line_options = line_options or {}
        scatter_options = scatter_options or {}
        label_options = label_options or {}
        title_options = title_options or {}

        line_options["color"] = line_options.get("color", "black")
        scatter_options["color"] = scatter_options.get("color", "red")

        x = pd.Series(np.linspace(lower, upper, 5000))
        reward = self.objective.__call__(x)
        fig, ax = plt.subplots()
        ax.plot(x, reward, **line_options)
        # TODO: validate dataframe
        if experiments is not None:
            x_data = experiments.loc[experiments[self.key].notna(), self.key].values
            ax.scatter(
                x_data,  # type: ignore
                self.objective.__call__(x_data),  # type: ignore
                **scatter_options,
            )
        ax.set_title("Objective %s" % self.key, **title_options)
        ax.set_ylabel("Objective", **label_options)
        ax.set_xlabel(self.key, **label_options)
        if plot_details:
            ax = self.objective.plot_details(ax=ax)
        return fig, ax

    def __str__(self) -> str:
        return "ContinuousOutputFeature"


# A helper constant for the default value of the weight parameter
FEATURE_ORDER = {
    ContinuousInput: 1,
    ContinuousDescriptorInput: 2,
    DiscreteInput: 3,
    CategoricalDescriptorInput: 4,
    CategoricalInput: 5,
    ContinuousOutput: 6,
}


## TODO: REMOVE THIS --> it is not needed!
def is_continuous(var: Feature) -> bool:
    """Checks if Feature is continous

    Args:
        var (Feature): Feature to be checked

    Returns:
        bool: True if continuous, else False
    """
    # TODO: generalize query via attribute continuousFeature (not existing yet!)
    if isinstance(var, ContinuousInput) or isinstance(var, ContinuousOutput):
        return True
    else:
        return False


# TODO: check lists of all features, possibly remove abstract classes
AnyFeature = Union[
    # InputFeature,
    # NumericalInputFeature,
    DiscreteInput,
    ContinuousInput,
    ContinuousDescriptorInput,
    CategoricalInput,
    CategoricalDescriptorInput,
    # OutputFeature,
    ContinuousOutput,
]
AnyInputFeature = Union[
    # InputFeature,
    # NumericalInputFeature,
    ContinuousInput,
    DiscreteInput,
    ContinuousDescriptorInput,
    CategoricalInput,
    CategoricalDescriptorInput,
]
AnyOutputFeature = ContinuousOutput
# Union[OutputFeature, ContinuousOutput,]


FeatureSequence = Union[List[AnyFeature], Tuple[AnyFeature]]


class Features(PydanticBaseModel):
    """Container of features, both input and output features are allowed.

    Attributes:
        features (List(Features)): list of the features.
    """

    features: FeatureSequence = Field(default_factory=lambda: [])

    def __iter__(self):
        return iter(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def __add__(self, other: Union[Sequence[AnyFeature], Features]):
        if isinstance(other, Features):
            other_feature_seq = other.features
        else:
            other_feature_seq = other
        new_feature_seq = list(itertools.chain(self.features, other_feature_seq))

        def is_feats_of_type(feats, ftype_collection, ftype_element):
            return isinstance(feats, ftype_collection) or (
                not isinstance(feats, Features)
                and (len(feats) > 0 and isinstance(feats[0], ftype_element))
            )

        def is_infeats(feats):
            return is_feats_of_type(feats, InputFeatures, InputFeature)

        def is_outfeats(feats):
            return is_feats_of_type(feats, OutputFeatures, OutputFeature)

        if is_infeats(self) and is_infeats(other):
            return InputFeatures(
                features=cast(Tuple[AnyInputFeature, ...], new_feature_seq)
            )
        if is_outfeats(self) and is_outfeats(other):
            return OutputFeatures(
                features=cast(Tuple[AnyOutputFeature, ...], new_feature_seq)
            )
        return Features(features=new_feature_seq)

    def get_by_key(self, key: str) -> AnyFeature:
        """Get a feature by its key.

        Args:
            key (str): Feature key of the feature of interest

        Returns:
            Feature: Feature of interest
        """
        return {f.key: f for f in self.features}[key]

    def get(
        self,
        includes: Union[Type, List[Type]] = AnyFeature,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> Features:
        """get features of the domain

        Args:
            includes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be returned. Defaults to Feature.
            excludes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.
            by_attribute (str, optional): If set it is filtered by the attribute specified in by `by_attribute`. Defaults to None.

        Returns:
            List[Feature]: List of features in the domain fitting to the passed requirements.
        """
        return self.__class__(
            features=sorted(
                filter_by_class(
                    self.features,
                    includes=includes,
                    excludes=excludes,
                    exact=exact,
                )
            )
        )

    def get_keys(
        self,
        includes: Union[Type, List[Type]] = AnyFeature,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> List[str]:
        """Method to get feature keys of the domain

        Args:
            includes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be returned. Defaults to Feature.
            excludes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[str]: List of feature keys fitting to the passed requirements.
        """
        return [
            f.key
            for f in self.get(
                includes=includes,
                excludes=excludes,
                exact=exact,
            )
        ]


Tnum_samples = conint(gt=0)


class InputFeatures(Features):
    """Container of input features, only input features are allowed.

    Attributes:
        features (List(InputFeatures)): list of the features.
    """

    features: Sequence[AnyInputFeature] = Field(default_factory=lambda: [])

    def get_fixed(self) -> "InputFeatures":
        """Gets all features in `self` that are fixed and returns them as new `InputFeatures` object.

        Returns:
            InputFeatures: Input features object containing only fixed features.
        """
        return InputFeatures(features=[feat for feat in self if feat.is_fixed()])  # type: ignore

    def get_free(self) -> "InputFeatures":
        """Gets all features in `self` that are not fixed and returns them as new `InputFeatures` object.

        Returns:
            InputFeatures: Input features object containing only non-fixed features.
        """
        return InputFeatures(features=[feat for feat in self if not feat.is_fixed()])  # type: ignore

    @validate_arguments
    def sample(
        self,
        n: Tnum_samples = 1,
        method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM,
    ) -> pd.DataFrame:
        """Draw sobol samples

        Args:
            n (int, optional): Number of samples, has to be larger than 0. Defaults to 1.
            method (SamplingMethodEnum, optional): Method to use, implemented methods are `UNIFORM`, `SOBOL` and `LHS`.
                Defaults to `UNIFORM`.

        Returns:
            pd.DataFrame: Dataframe containing the samples.
        """
        if method == SamplingMethodEnum.UNIFORM:
            return self.validate_inputs(
                pd.concat([feat.sample(n) for feat in self.get(InputFeature)], axis=1)  # type: ignore
            )
        free_features = self.get_free()
        if method == SamplingMethodEnum.SOBOL:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = Sobol(len(free_features)).random(n)
        else:
            X = LatinHypercube(len(free_features)).random(n)
        res = []
        for i, feat in enumerate(free_features):
            if isinstance(feat, ContinuousInput):
                x = feat.from_unit_range(X[:, i])
            elif isinstance(feat, (DiscreteInput, CategoricalInput)):
                if isinstance(feat, DiscreteInput):
                    levels = feat.values
                else:
                    levels = feat.get_allowed_categories()
                bins = np.linspace(0, 1, len(levels) + 1)
                idx = np.digitize(X[:, i], bins) - 1
                x = np.array(levels)[idx]
            else:
                raise (ValueError(f"Unknown input feature with key {feat.key}"))
            res.append(pd.Series(x, name=feat.key))
        samples = pd.concat(res, axis=1)
        for feat in self.get_fixed():
            samples[feat.key] = feat.fixed_value()[0]  # type: ignore
        return self.validate_inputs(samples)[self.get_keys(InputFeature)]

    # validate candidates, TODO rename and tidy up
    def validate_inputs(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Validate a pandas dataframe with input feature values.

        Args:
            inputs (pd.Dataframe): Inputs to validate.

        Raises:
            ValueError: Raises a Valueerror if a feature based validation raises an exception.

        Returns:
            pd.Dataframe: Validated dataframe
        """
        for feature in self:
            if feature.key not in inputs:
                raise ValueError(f"no col for input feature `{feature.key}`")
            feature.validate_candidental(inputs[feature.key])  # type: ignore
        return inputs

    def validate_experiments(
        self, experiments: pd.DataFrame, strict=False
    ) -> pd.DataFrame:
        for feature in self:
            if feature.key not in experiments:
                raise ValueError(f"no col for input feature `{feature.key}`")
            feature.validate_experimental(experiments[feature.key], strict=strict)  # type: ignore
        return experiments

    def get_categorical_combinations(
        self,
        include: Type[Feature] = InputFeature,
        exclude: Optional[Type[InputFeature]] = None,
    ):
        """get a list of tuples pairing the feature keys with a list of valid categories

        Args:
            include (Feature, optional): Features to be included. Defaults to InputFeature.
            exclude (Feature, optional): Features to be excluded, e.g. subclasses of the included features. Defaults to None.

        Returns:
            List[(str, List[str])]: Returns a list of tuples pairing the feature keys with a list of valid categories (str)
        """
        features = [
            f
            for f in self.get(includes=include, excludes=exclude)
            if isinstance(f, CategoricalInput) and not f.is_fixed()
        ]
        list_of_lists = [
            [(f.key, cat) for cat in f.get_allowed_categories()] for f in features
        ]
        return list(itertools.product(*list_of_lists))

    # transformation related methods
    def _get_transform_info(
        self, specs: TInputTransformSpecs
    ) -> Tuple[Dict[str, Tuple[int]], Dict[str, Tuple[str]]]:
        """Generates two dictionaries. The first one specifies which key is mapped to
        which column indices when applying `transform`. The second one specifies
        which key is mapped to which transformed keys.

        Args:
            specs (TInputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            Dict[str, Tuple[int]]: Dictionary mapping feature keys to column indices.
            Dict[str, Tuple[str]]: Dictionary mapping feature keys to transformed feature
                keys.
        """
        self._validate_transform_specs(specs)
        features2idx = {}
        features2names = {}
        counter = 0
        for _, feat in enumerate(self.get()):
            if feat.key not in specs.keys():
                features2idx[feat.key] = (counter,)
                features2names[feat.key] = (feat.key,)
                counter += 1
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.categories))) + counter).tolist()
                )
                features2names[feat.key] = tuple(
                    [f"{feat.key}{_CAT_SEP}{c}" for c in feat.categories]
                )
                counter += len(feat.categories)
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                features2idx[feat.key] = (counter,)
                features2names[feat.key] = (feat.key,)
                counter += 1
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.categories) - 1)) + counter).tolist()
                )
                features2names[feat.key] = tuple(
                    [f"{feat.key}{_CAT_SEP}{c}" for c in feat.categories[1:]]
                )
                counter += len(feat.categories) - 1
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.descriptors))) + counter).tolist()
                )
                features2names[feat.key] = tuple(
                    [f"{feat.key}{_CAT_SEP}{d}" for d in feat.descriptors]
                )
                counter += len(feat.descriptors)
        return features2idx, features2names

    def transform(
        self, experiments: pd.DataFrame, specs: TInputTransformSpecs
    ) -> pd.DataFrame:
        """Transform a dataframe to the represenation specified in `specs`.

        Currently only input categoricals are supported.

        Args:
            experiments (pd.DataFrame): Data dataframe to be transformed.
            specs (TInputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            pd.DataFrame: Transformed dataframe. Only input features are included.
        """
        specs = self._validate_transform_specs(specs)
        transformed = []
        for feat in self.get():
            s = experiments[feat.key]
            if feat.key not in specs.keys():
                transformed.append(s)
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_onehot_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_ordinal_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_dummy_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                transformed.append(feat.to_descriptor_encoding(s))
        return pd.concat(transformed, axis=1)

    def inverse_transform(
        self, experiments: pd.DataFrame, specs: TInputTransformSpecs
    ) -> pd.DataFrame:
        """Transform a dataframe back to the original representations.

        The original applied transformation has to be provided via the specs dictionary.
        Currently only input categoricals are supported.

        Args:
            experiments (pd.DataFrame): Transformed data dataframe.
            specs (TInputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            pd.DataFrame: Back transformed dataframe. Only input features are included.
        """
        self._validate_transform_specs(specs=specs)
        transformed = []
        for feat in self.get():
            if feat.key not in specs.keys():
                transformed.append(experiments[feat.key])
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.from_onehot_encoding(experiments))
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.from_ordinal_encoding(experiments[feat.key]))
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.from_dummy_encoding(experiments))
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                transformed.append(feat.from_descriptor_encoding(experiments))
        return pd.concat(transformed, axis=1)

    def _validate_transform_specs(self, specs: TInputTransformSpecs):
        """Checks the validity of the transform specs .

        Args:
            specs (TInputTransformSpecs): Transform specs to be validated.
        """
        # first check that the keys in the specs dict are correct also correct feature keys
        if len(set(specs.keys()) - set(self.get_keys(CategoricalInput))) > 0:
            raise ValueError("Unknown features specified in transform specs.")
        # next check that all values are of type CategoricalEncodingEnum
        if not (
            all([isinstance(enc, CategoricalEncodingEnum) for enc in specs.values()])
        ):
            raise ValueError("Unknown transform specified.")
        # next check that only Categoricalwithdescriptor have the value DESCRIPTOR
        descriptor_keys = [
            key
            for key, value in specs.items()
            if value == CategoricalEncodingEnum.DESCRIPTOR
        ]
        if (
            len(set(descriptor_keys) - set(self.get_keys(CategoricalDescriptorInput)))
            > 0
        ):
            raise ValueError("Wrong features types assigned to DESCRIPTOR transform.")
        return specs

    def get_bounds(
        self,
        specs: TInputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[float], List[float]]:
        """Returns the boundaries of the optimization problem based on the transformations
        defined in the  `specs` dictionary.

        Args:
            specs (TInputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.
            experiments (Optional[pd.DataFrame], optional): Dataframe with input features.
                If provided the real feature bounds are returned based on both the opt.
                feature bounds and the extreme points in the dataframe. Defaults to None,

        Raises:
            ValueError: If a feature type is not known.
            ValueError: If no transformation is provided for a categorical feature.

        Returns:
            Tuple[List[float], List[float]]: list with lower bounds, list with upper bounds.
        """
        self._validate_transform_specs(specs=specs)

        lower = []
        upper = []

        for feat in self.get():
            l, u = feat.get_bounds(  # type: ignore
                transform_type=specs.get(feat.key),  # type: ignore
                values=experiments[feat.key] if experiments is not None else None,
            )
            lower += l
            upper += u
        return lower, upper


class OutputFeatures(Features):
    """Container of output features, only output features are allowed.

    Attributes:
        features (List(OutputFeatures)): list of the features.
    """

    features: Sequence[AnyOutputFeature] = Field(default_factory=lambda: [])

    def get_by_objective(
        self,
        includes: Union[
            List[Type[AnyAbstractObjective]],
            Type[AnyAbstractObjective],
            Type[Objective],
        ] = Objective,
        excludes: Union[
            List[Type[AnyAbstractObjective]],
            Type[AnyAbstractObjective],
            None,
        ] = None,
        exact: bool = False,
    ) -> "OutputFeatures":
        """Get output features filtered by the type of the attached objective.

        Args:
            includes (Union[List[TObjective], TObjective], optional): Objective class or list of objective classes
                to be returned. Defaults to Objective.
            excludes (Union[List[TObjective], TObjective, None], optional): Objective class or list of specific objective classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact classes listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[AnyOutputFeature]: List of output features fitting to the passed requirements.
        """
        if len(self.features) == 0:
            return OutputFeatures(features=[])
        else:
            # TODO: why only continuous output?
            return OutputFeatures(
                features=sorted(
                    filter_by_attribute(
                        self.get(ContinuousOutput).features,
                        lambda of: of.objective,
                        includes,
                        excludes,
                        exact,
                    )
                )
            )

    def get_keys_by_objective(
        self,
        includes: Union[
            List[Type[AnyAbstractObjective]],
            Type[AnyAbstractObjective],
            Type[Objective],
        ] = Objective,
        excludes: Union[
            List[Type[AnyAbstractObjective]], Type[AnyAbstractObjective], None
        ] = None,
        exact: bool = False,
    ) -> List[str]:
        """Get keys of output features filtered by the type of the attached objective.

        Args:
            includes (Union[List[TObjective], TObjective], optional): Objective class or list of objective classes
                to be returned. Defaults to Objective.
            excludes (Union[List[TObjective], TObjective, None], optional): Objective class or list of specific objective classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact classes listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[str]: List of output feature keys fitting to the passed requirements.
        """
        return [f.key for f in self.get_by_objective(includes, excludes, exact)]

    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the objective for every

        Args:
            experiments (pd.DataFrame): Experiments for which the objectives should be evaluated.

        Returns:
            pd.DataFrame: Objective values for the experiments of interest.
        """
        return pd.concat(
            [
                feat.objective(experiments[[feat.key]])  # type: ignore
                for feat in self.features
                if feat.objective is not None
            ],
            axis=1,
        )

    def preprocess_experiments_one_valid_output(
        self,
        output_feature_key: str,
        experiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of the provided output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_key (str): The feature based on which non-valid entries rows are removed

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the specific feature are included
        """
        clean_exp = experiments.loc[
            (experiments["valid_%s" % output_feature_key] == 1)
            & (experiments[output_feature_key].notna())
        ]

        return clean_exp

    def preprocess_experiments_all_valid_outputs(
        self,
        experiments: pd.DataFrame,
        output_feature_keys: Optional[List] = None,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of all output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_keys (Optional[List], optional): List of output feature keys which should be considered for removal of invalid values. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the selected features are included
        """
        if (output_feature_keys is None) or (len(output_feature_keys) == 0):
            output_feature_keys = self.get_keys(OutputFeature)

        clean_exp = experiments.query(
            " & ".join(["(`valid_%s` > 0)" % key for key in output_feature_keys])
        )
        clean_exp = clean_exp.dropna(subset=output_feature_keys)

        return clean_exp

    def preprocess_experiments_any_valid_output(
        self, experiments: pd.DataFrame
    ) -> pd.DataFrame:
        """Method to get a dataframe where at least one output feature has a valid entry

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Returns:
            pd.DataFrame: Dataframe with all experiments where at least one output feature has a valid entry
        """

        output_feature_keys = self.get_keys(OutputFeature)

        # clean_exp = experiments.query(" or ".join(["(valid_%s > 0)" % key for key in output_feature_keys]))
        # clean_exp = clean_exp.query(" or ".join(["%s.notna()" % key for key in output_feature_keys]))

        assert experiments is not None
        clean_exp = experiments.query(
            " or ".join(
                [
                    "((`valid_%s` >0) & `%s`.notna())" % (key, key)
                    for key in output_feature_keys
                ]
            )
        )

        return clean_exp
