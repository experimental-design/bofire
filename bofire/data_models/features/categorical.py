from typing import Annotated, ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.feature import (
    Input,
    Output,
    TTransform,
    get_encoded_name,
)
from bofire.data_models.objectives.api import AnyCategoricalObjective
from bofire.data_models.types import CategoryVals


class CategoricalInput(Input):
    """Base class for all categorical input features.

    Attributes:
        categories (List[str]): Names of the categories.
        allowed (List[bool]): List of bools indicating if a category is allowed within the optimization.

    """

    type: Literal["CategoricalInput"] = "CategoricalInput"  # type: ignore
    # order_id: ClassVar[int] = 5
    order_id: ClassVar[int] = 7

    categories: CategoryVals
    allowed: Optional[Annotated[List[bool], Field(min_length=2)]] = Field(
        default=None,
        validate_default=True,
    )

    @field_validator("allowed")
    @classmethod
    def generate_allowed(cls, allowed, info):
        """Generates the list of allowed categories if not provided."""
        if allowed is None and "categories" in info.data.keys():
            return [True for _ in range(len(info.data["categories"]))]
        return allowed

    @model_validator(mode="after")
    def validate_categories_fitting_allowed(self):
        if len(self.allowed) != len(self.categories):  # type: ignore
            raise ValueError("allowed must have same length as categories")
        if sum(self.allowed) == 0:  # type: ignore
            raise ValueError("no category is allowed")
        return self

    @staticmethod
    def valid_transform_types() -> List[CategoricalEncodingEnum]:  # type: ignore
        return [
            CategoricalEncodingEnum.ONE_HOT,
            CategoricalEncodingEnum.DUMMY,
            CategoricalEncodingEnum.ORDINAL,
        ]

    def is_fixed(self) -> bool:
        """Returns True if there is only one allowed category.

        Returns:
            [bool]: True if there is only one allowed category

        """
        if self.allowed is None:
            return False
        return sum(self.allowed) == 1

    def fixed_value(
        self,
        transform_type: Optional[TTransform] = None,
    ) -> Union[List[str], List[float], None]:
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None

        """
        if self.is_fixed():
            val = self.get_allowed_categories()[0]
            if transform_type is None:
                return [val]
            if transform_type == CategoricalEncodingEnum.ONE_HOT:
                return self.to_onehot_encoding(pd.Series([val])).values[0].tolist()
            if transform_type == CategoricalEncodingEnum.DUMMY:
                return self.to_dummy_encoding(pd.Series([val])).values[0].tolist()
            if transform_type == CategoricalEncodingEnum.ORDINAL:
                return self.to_ordinal_encoding(pd.Series([val])).tolist()
            raise ValueError(
                f"Unkwon transform type {transform_type} for categorical input {self.key}",
            )
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
        self,
        values: pd.Series,
        strict: bool = False,
    ) -> pd.Series:
        """Method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurrence of fixed features in the dataset should be considered or not. Defaults to False.

        Raises:
            ValueError: when an entry is not in the list of allowed categories
            ValueError: when there is no variation in a feature provided by the experimental data

        Returns:
            pd.Series: A dataFrame with experiments

        """
        values = values.map(str)
        if sum(values.isin(self.categories)) != len(values):
            raise ValueError(
                f"invalid values for `{self.key}`, allowed are: `{self.categories}`",
            )
        if strict:
            possible_categories = self.get_possible_categories(values)
            if len(possible_categories) != len(self.categories):
                raise ValueError(
                    f"Categories {list(set(self.categories)-set(possible_categories))} of feature {self.key} not used. Remove them.",
                )
        return values

    def is_fulfilled(self, values: pd.Series) -> pd.Series:
        """Method to check if the values are all allowed categories.

        Args:
            values: A series with values for the input feature.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.
        """
        return values.isin(self.get_allowed_categories())

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when not all values for a feature are one of the allowed categories

        Returns:
            pd.Series: The passed dataFrame with candidates

        """
        values = values.map(str)
        if sum(values.isin(self.get_allowed_categories())) != len(values):
            raise ValueError(
                f"not all values of input feature `{self.key}` are a valid allowed category from {self.get_allowed_categories()}",
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
        return sorted(set(list(set(values.tolist())) + self.get_allowed_categories()))

    def to_onehot_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a one-hot encoding.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: One-hot transformed data frame.

        """
        return pd.DataFrame(
            {get_encoded_name(self.key, c): values == c for c in self.categories},
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
        cat_cols = [get_encoded_name(self.key, c) for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}.",
            )
        s = values[cat_cols].idxmax(1).str[(len(self.key) + 1) :]
        s.name = self.key
        return s

    def to_dummy_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a dummy-hot encoding, dropping the first categorical level.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: Dummy-hot transformed data frame.

        """
        return pd.DataFrame(
            {get_encoded_name(self.key, c): values == c for c in self.categories[1:]},
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
        cat_cols = [get_encoded_name(self.key, c) for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols[1:]]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols[1:]}.",
            )
        values = values.copy()
        values[cat_cols[0]] = 1 - values[cat_cols[1:]].sum(axis=1)
        s = values[cat_cols].idxmax(1).str[(len(self.key) + 1) :]
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

    def sample(self, n: int, seed: Optional[int] = None) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.
            seed (int, optional): random seed. Defaults to None.

        Returns:
            pd.Series: drawn samples.

        """
        return pd.Series(
            name=self.key,
            data=np.random.default_rng(seed=seed).choice(
                self.get_allowed_categories(),
                n,
            ),
        )

    def get_bounds(  # type: ignore
        self,
        transform_type: TTransform,
        values: Optional[pd.Series] = None,
        reference_value: Optional[str] = None,
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
                f"Invalid descriptor transform for categorical {self.key}.",
            )
        raise ValueError(
            f"Invalid transform_type {transform_type} provided for categorical {self.key}.",
        )

    def __str__(self) -> str:
        """Returns the number of categories as str

        Returns:
            str: Number of categories

        """
        return f"{len(self.categories)} categories"


class CategoricalOutput(Output):
    type: Literal["CategoricalOutput"] = "CategoricalOutput"  # type: ignore
    order_id: ClassVar[int] = 10

    categories: CategoryVals
    objective: AnyCategoricalObjective

    @model_validator(mode="after")
    def validate_objective_categories(self):
        """Validates that objective categories match the output categories

        Raises:
            ValueError: when categories do not match objective categories

        Returns:
            self

        """
        if self.objective.categories != self.categories:
            raise ValueError("categories must match to objective categories")
        return self

    def __call__(self, values: pd.Series, values_adapt: pd.Series) -> pd.Series:  # type: ignore
        if self.objective is None:
            return pd.Series(
                data=[np.nan for _ in range(len(values))],
                index=values.index,
                name=values.name,
            )
        return self.objective(values, values_adapt)  # type: ignore

    def validate_experimental(self, values: pd.Series) -> pd.Series:
        values = values.map(str)
        if sum(values.isin(self.categories)) != len(values):
            raise ValueError(
                f"invalid values for `{self.key}`, allowed are: `{self.categories}`",
            )
        return values

    def __str__(self) -> str:
        return "CategoricalOutputFeature"
