from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field, validator
from pydantic.class_validators import root_validator
from pydantic.types import conlist
from scipy.stats.qmc import Sobol

from bofire.domain.objectives import MaximizeObjective, Objective
from bofire.domain.util import (
    BaseModel,
    KeyModel,
    filter_by_attribute,
    filter_by_class,
    is_numeric,
    name2key,
)


class Feature(KeyModel):
    """The base class for all features."""

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

    def to_config(self) -> Dict:
        """Generate serialized version of the feature.

        Returns:
            Dict: Serialized version of the feature as dictionary.
        """
        return {
            "type": self.__class__.__name__,
            **self.dict(),
        }

    @staticmethod
    def from_config(config: Dict) -> "Feature":
        """Generate objective out of serialized version.

        Args:
            config (Dict): Serialized version of a objective

        Returns:
            Objective: Instantiated objective of the type specified in the `config`.
        """
        input_mapper = {
            "ContinuousInput": ContinuousInput,
            "DiscreteInput": DiscreteInput,
            "CategoricalInput": CategoricalInput,
            "CategoricalDescriptorInput": CategoricalDescriptorInput,
            "ContinuousDescriptorInput": ContinuousDescriptorInput,
        }
        output_mapper = {
            "ContinuousOutput": ContinuousOutput,
        }
        if config["type"] in input_mapper.keys():
            return input_mapper[config["type"]](**config)
        else:
            if "objective" in config.keys():
                obj = Objective.from_config(config=config["objective"])
            else:
                obj = None
            return output_mapper[config["type"]](key=config["key"], objective=obj)


TFeature = TypeVar("TFeature", bound=Feature)


class InputFeature(Feature):
    """Base class for all input features."""

    @abstractmethod
    def is_fixed() -> bool:
        """Indicates if a variable is set to a fixed value.

        Returns:
            bool: True if fixed, els False.
        """
        pass

    @abstractmethod
    def fixed_value() -> Union[None, str, float]:
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


class NumericalInputFeature(InputFeature):
    """Abstracht base class for all numerical (ordinal) input features."""

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
            lower, upper = self.get_real_feature_bounds(values)
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

    def fixed_value(self):
        """Method to get the value to which the feature is fixed

        Returns:
            Float: Return the feature value or None if the feature is not fixed.
        """
        if self.is_fixed():
            return self.lower_bound  # type: ignore
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
            lower, upper = self.get_real_feature_bounds(values)
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

    def get_real_feature_bounds(
        self, values: Union[pd.Series, np.ndarray]
    ) -> Tuple[float, float]:
        """Method to extract the feature boundaries from the provided experimental data

        Args:
            values (pd.Series): Experimental data

        Returns:
            (float, float): Returns lower and upper bound based on the passed data
        """
        lower = min(self.lower_bound, values.min())  # type: ignore
        upper = max(self.upper_bound, values.max())  # type: ignore
        return lower, upper


class ContinuousInput(NumericalInputFeature):
    """Base class for all continuous input features.

    Attributes:
        lower_bound (float): Lower bound of the feature in the optimization.
        upper_bound (float): Upper bound of the feature in the optimization.
    """

    lower_bound: float
    upper_bound: float

    @root_validator(pre=False)
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
        noise = 10e-8
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


class DiscreteInput(NumericalInputFeature):
    """Feature with discretized ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the discretized allowed values during the optimization.
    """

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

    @root_validator(pre=False)
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

    @root_validator(pre=False)
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

    def is_fixed(self):
        """Returns True if there is only one allowed category.

        Returns:
            [bool]: True if there is only one allowed category
        """
        if self.allowed is None:
            return False
        return sum(self.allowed) == 1

    def fixed_value(self):
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None
        """
        if self.is_fixed():
            return self.get_allowed_categories()[0]
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
        calues (List[List[float]]): List of lists representing the descriptor values.
    """

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

    def get_real_descriptor_bounds(self, values) -> pd.DataFrame:
        """Method to generate a dataFrame as tabular overview of lower and upper bounds of the descriptors (excluding non-allowed descriptors)

        Args:
            values (pd.Series): The categories present in the passed data for the considered feature

        Returns:
            pd.Series: Tabular overview of lower and upper bounds of the descriptors
        """
        df = self.to_df().loc[self.get_possible_categories(values)]
        data = {
            "lower": [min(df[desc].tolist()) for desc in self.descriptors],
            "upper": [max(df[desc].tolist()) for desc in self.descriptors],
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.descriptors)

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
            bounds = self.get_real_descriptor_bounds(values)
            for desc in self.descriptors:
                if bounds.loc["lower", desc] == bounds.loc["upper", desc]:
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


class OutputFeature(Feature):
    """Base class for all output features.

    Attributes:
        key(str): Key of the Feature.
    """

    objective: Optional[Objective]


class ContinuousOutput(OutputFeature):
    """The base class for a continuous output feature

    Attributes:
        objective (objective, optional): objective of the feature indicating in which direction it should be optimzed. Defaults to `MaximizeObjective`.
    """

    objective: Optional[Objective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0)
    )

    def to_config(self) -> Dict:
        """Generate serialized version of the feature.

        Returns:
            Dict: Serialized version of the feature as dictionary.
        """
        config: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "key": self.key,
        }
        if self.objective is not None:
            config["objective"] = self.objective.to_config()
        return config

    def plot(
        self,
        lower: float,
        upper: float,
        df_data: Optional[pd.DataFrame] = None,
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
            df_data (Optional[pd.DataFrame], optional): If provided, scatter also the historical data in the plot. Defaults to None.
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

        x = pd.DataFrame(np.linspace(lower, upper, 5000))
        reward = self.objective.__call__(x)
        fig, ax = plt.subplots()
        ax.plot(x, reward, **line_options)
        # TODO: validate dataframe
        if df_data is not None:
            x_data = df_data.loc[df_data[self.key].notna(), self.key].values
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
    CategoricalInput: 4,
    CategoricalDescriptorInput: 5,
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


TFeature = TypeVar("TFeature", bound=Feature)


class Features(BaseModel):
    """Container of features, both input and output features are allowed.

    Attributes:
        features (List(Features)): list of the features.
    """

    features: List[Feature] = Field(default_factory=lambda: [])

    def to_config(self) -> Dict:
        """Serialize the features container.

        Returns:
            Dict: serialized features container
        """
        return {
            "type": "general",
            "features": [feat.to_config() for feat in self.features],
        }

    @staticmethod
    def from_config(config: Dict) -> "Features":
        """Instantiates a `Feature` object from a dictionary created by the `to_config`method.

        Args:
            config (Dict): Serialized features dictionary

        Returns:
            Features: instantiated features object
        """
        if config["type"] == "inputs":
            return InputFeatures(
                features=[
                    cast(InputFeature, Feature.from_config(feat))
                    for feat in config["features"]
                ]
            )
        if config["type"] == "outputs":
            return OutputFeatures(
                features=[
                    cast(OutputFeature, Feature.from_config(feat))
                    for feat in config["features"]
                ]
            )
        if config["type"] == "general":
            return Features(
                features=[Feature.from_config(feat) for feat in config["features"]]
            )
        else:
            raise ValueError(f"Unknown type {config['type']} provided.")

    def __iter__(self):
        return iter(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def __add__(self, other):
        if type(self) != type(other):
            return Features(features=self.features + other.features)
        if type(other) == InputFeatures:
            return InputFeatures(features=self.features + other.features)
        if type(other) == OutputFeatures:
            return OutputFeatures(features=self.features + other.features)
        return Features(features=self.features + other.features)

    def remove(self, feature: Feature):
        self.features.remove(feature)

    def add(self, feature: Feature):
        """Add a feature to the container.

        Args:
            feature (Feature): Feature to be added.
        """
        assert isinstance(feature, Feature)
        self.features.append(feature)

    def get_by_key(self, key: str) -> Feature:
        """Get a feature by its key.

        Args:
            key (str): Feature key of the feature of interest

        Returns:
            Feature: Feature of interest
        """
        return {f.key: f for f in self.features}[key]

    def get(
        self,
        includes: Union[Type, List[Type]] = Feature,
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
        includes: Union[Type, List[Type]] = Feature,
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


class InputFeatures(Features):
    """Container of input features, only input features are allowed.

    Attributes:
        features (List(InputFeatures)): list of the features.
    """

    features: List[InputFeature] = Field(default_factory=lambda: [])

    def to_config(self) -> Dict:
        return {
            "type": "inputs",
            "features": [feat.to_config() for feat in self.features],
        }

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

    def sample_uniform(self, n: int = 1) -> pd.DataFrame:
        """Draw uniformly random samples

        Args:
            n (int, optional): Number of samples. Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe containing the samples.
        """
        return self.validate_inputs(
            pd.concat([feat.sample(n) for feat in self.get(InputFeature)], axis=1)  # type: ignore
        )

    def sample_sobol(self, n: int) -> pd.DataFrame:
        """Draw uniformly random samples

        Args:
            n (int, optional): Number of samples. Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe containing the samples.
        """
        free_features = self.get_free()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = Sobol(len(free_features)).random(n)
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
            samples[feat.key] = feat.fixed_value()  # type: ignore
        return self.validate_inputs(samples)[self.get_keys(InputFeature)]

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

    def add(self, feature: InputFeature):
        """Add a input feature to the container.

        Args:
            feature (InputFeature): InputFeature to be added.
        """
        assert isinstance(feature, InputFeature)
        self.features.append(feature)

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


class OutputFeatures(Features):
    """Container of output features, only output features are allowed.

    Attributes:
        features (List(OutputFeatures)): list of the features.
    """

    features: List[OutputFeature] = Field(default_factory=lambda: [])

    def to_config(self) -> Dict:
        return {
            "type": "outputs",
            "features": [feat.to_config() for feat in self.features],
        }

    @validator("features", pre=True)
    def validate_output_features(cls, v, values):
        for feat in v:
            if not isinstance(feat, OutputFeature):
                raise ValueError
        return v

    def add(self, feature: OutputFeature):
        """Add a output feature to the container.

        Args:
            feature (OutputFeature): OutputFeature to be added.
        """
        assert isinstance(feature, OutputFeature)
        self.features.append(feature)

    def get_by_objective(
        self,
        includes: Union[List[Type[Objective]], Type[Objective]] = Objective,
        excludes: Union[List[Type[Objective]], Type[Objective], None] = None,
        exact: bool = False,
    ) -> "OutputFeatures":
        """Get output features filtered by the type of the attached objective.

        Args:
            includes (Union[List[TObjective], TObjective], optional): Objective class or list of objective classes
                to be returned. Defaults to Objective.
            excludes (Union[List[TObjective], TObjective, None], optional): Objective class or list of specific objective classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact classes listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[OutputFeature]: List of output features fitting to the passed requirements.
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
        includes: Union[List[Type[Objective]], Type[Objective]] = Objective,
        excludes: Union[List[Type[Objective]], Type[Objective], None] = None,
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
                feat.objective(experiments[[feat.key]])
                for feat in self.features
                if feat.objective is not None
            ],
            axis=1,
        )
