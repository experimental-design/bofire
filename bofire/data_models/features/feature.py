from abc import abstractmethod
from typing import ClassVar, List, Optional, Tuple, Union

import pandas as pd

from bofire.data_models.base import BaseModel
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.molfeatures.api import AnyMolFeatures
from bofire.data_models.surrogates.scaler import ScalerEnum


TTransform = Union[CategoricalEncodingEnum, ScalerEnum, AnyMolFeatures]


class Feature(BaseModel):
    """The base class for all features."""

    type: str
    key: str
    order_id: ClassVar[int] = -1

    def __lt__(self, other) -> bool:
        """Method to compare two models to get them in the desired order.
        Return True if other is larger than self, else False. (see FEATURE_ORDER)

        Args:
            other: The other class to compare to self

        Returns:
            bool: True if the other class is larger than self, else False

        """
        order_self = self.order_id
        order_other = other.order_id
        if order_self == order_other:
            return self.key < other.key
        return order_self < order_other


class Input(Feature):
    """Base class for all input features."""

    @staticmethod
    @abstractmethod
    def valid_transform_types() -> List[Union[CategoricalEncodingEnum, AnyMolFeatures]]:
        pass

    @abstractmethod
    def is_fixed(self) -> bool:
        """Indicates if a variable is set to a fixed value.

        Returns:
            bool: True if fixed, els False.

        """

    @abstractmethod
    def is_fulfilled(self, values: pd.Series) -> pd.Series:
        """Method to check if the input feature is fulfilled by the values.

        Args:
            values: A series with values for the input feature.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.

        """
        pass

    @abstractmethod
    def fixed_value(
        self,
        transform_type: Optional[TTransform] = None,
    ) -> Union[None, List[str], List[float]]:
        """Method to return the fixed value in case of a fixed feature.

        Returns:
            Union[None,str,float]: None in case the feature is not fixed, else the fixed value.

        """

    @abstractmethod
    def validate_experimental(
        self,
        values: pd.Series,
        strict: bool = False,
    ) -> pd.Series:
        """Abstract method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurrence of fixed features in the dataset should be considered or not. Defaults to False.

        Returns:
            pd.Series: The passed dataFrame with experiments

        """

    @abstractmethod
    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Abstract method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Returns:
            pd.Series: The passed dataFrame with candidates

        """

    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None) -> pd.Series:
        """Sample a series of allowed values.

        Args:
            n (int): Number of samples
            seed (int, optional): random seed. Defaults to None.

        Returns:
            pd.Series: Sampled values.

        """

    @abstractmethod
    def get_bounds(
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
        reference_value: Optional[Union[float, str]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Returns the bounds of an input feature depending on the requested transform type.

        Args:
            transform_type (Optional[TTransform], optional): The requested transform type. Defaults to None.
            values (Optional[pd.Series], optional): If values are provided the bounds are returned taking
                the most extreme values for the feature into account. Defaults to None.
            reference_value (Optional[float], optional): If a reference value is provided, then the local bounds based
                on a local search region are provided. Currently only supported for continuous inputs. For more
                details, it is referred to https://www.merl.com/publications/docs/TR2023-057.pdf.

        Returns:
            Tuple[List[float], List[float]]: List of lower bound values, list of upper bound values.

        """


class Output(Feature):
    """Base class for all output features.

    Attributes:
        key(str): Key of the Feature.

    """

    @abstractmethod
    def __call__(self, values: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def validate_experimental(self, values: pd.Series) -> pd.Series:
        """Abstract method to validate the experimental Series

        Args:
            values (pd.Series): A dataFrame with values for the outcome

        Returns:
            pd.Series: The passed dataFrame with experiments

        """


def is_numeric(s: Union[pd.Series, pd.DataFrame]) -> bool:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").notnull().all()  # type: ignore
    return s.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all()).all()  # type: ignore


def is_categorical(s: pd.Series, categories: List[str]):
    return sum(s.isin(categories)) == len(s)


def get_encoded_name(feature_key: str, option_name: str) -> str:
    """Get the name of the encoded column. Option could be the category or the descriptor name."""
    return f"{feature_key}_{option_name}"
