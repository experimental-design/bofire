from abc import abstractmethod
from typing import Dict, List, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import Field, validator
from pydantic.class_validators import root_validator
from pydantic.types import conlist

from bofire.domain.util import BaseModel, filter_by_class


class Constraint(BaseModel):
    """Abstract base class to define constraints on the optimization space."""

    @abstractmethod
    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Abstract method to check if a constraint is fulfilled for all the rows of the provided dataframe.

        Args:
            experiments (pd.DataFrame): Dataframe to check constraint fulfillment.

        Returns:
            bool: True if fulfilled else False
        """
        pass

    @abstractmethod
    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.
        """
        pass

    def to_config(self) -> Dict:
        """Generate serialized version of the constraint.

        Returns:
            Dict: Serialized version of the constraint as dictionary.
        """
        return {
            "type": self.__class__.__name__,
            **self.dict(),
        }

    @staticmethod
    def from_config(config: Dict) -> "Constraint":
        """Generate constraint out of serialized version.

        Args:
            config (Dict): Serialized version of a constraint

        Returns:
            Constraint: Instaniated constraint of the type specified in the `config`.
        """
        mapper = {
            "LinearEqualityConstraint": LinearEqualityConstraint,
            "LinearInequalityConstraint": LinearInequalityConstraint,
            "NChooseKConstraint": NChooseKConstraint,
            "NonlinearEqualityConstraint": NonlinearEqualityConstraint,
            "NonlinearInqualityConstraint": NonlinearInqualityConstraint,
        }
        return mapper[config["type"]](**config)


TFeatureKeys = conlist(item_type=str, min_items=2)
TCoefficients = conlist(item_type=float, min_items=2)


class LinearConstraint(Constraint):
    """Abstract base class for linear equality and inequality constraints.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    features: TFeatureKeys
    coefficients: TCoefficients
    rhs: float

    @validator("features")
    def validate_features_unique(cls, features):
        """Validate that feature keys are unique."""
        if len(features) != len(set(features)):
            raise ValueError("features must be unique")
        return features

    @root_validator(pre=False)
    def validate_list_lengths(cls, values):
        """Validate that length of the feature and coefficient lists have the same length."""
        if len(values["features"]) != len(values["coefficients"]):
            raise ValueError(
                f'must provide same number of features and coefficients, got {len(values["features"])} != {len(values["coefficients"])}'
            )
        return values

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return (
            experiments[self.features] @ self.coefficients - self.rhs
        ) / np.linalg.norm(self.coefficients)

    # def lhs(self, df_data: pd.DataFrame) -> float:
    #     """Evaluate the left-hand side of the constraint on each row of a dataframe

    #     Args:
    #         df_data (pd.DataFrame): Dataframe on which the left-hand side should be evaluated.

    #     Returns:
    #         np.array: 1-dim array with left-hand side of each row of the provided dataframe.
    #     """
    #     cols = self.features
    #     coefficients = self.coefficients
    #     return np.sum(df_data[cols].values * np.array(coefficients), axis=1)

    def __str__(self) -> str:
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return " + ".join(
            [f"{self.coefficients[i]} * {feat}" for i, feat in enumerate(self.features)]
        )


class LinearEqualityConstraint(LinearConstraint):
    """Linear equality constraint of the form `coefficients * x = rhs`.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    # def is_fulfilled(self, experiments: pd.DataFrame, complete: bool) -> bool:
    #     """Check if the linear equality constraint is fulfilled for all the rows of the provided dataframe.

    #     Args:
    #         df_data (pd.DataFrame): Dataframe to evaluate constraint on.

    #     Returns:
    #         bool: True if fulfilled else False.
    #     """
    #     fulfilled = np.isclose(self(experiments), 0)
    #     if complete:
    #         return fulfilled.all()
    #     else:
    #         pd.Series(fulfilled, index=experiments.index)

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(experiments), 0), index=experiments.index)

    def __str__(self) -> str:
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return super().__str__() + f" = {self.rhs}"


class LinearInequalityConstraint(LinearConstraint):
    """Linear inequality constraint of the form `coefficients * x <= rhs`.

    To instantiate a constraint of the form `coefficients * x >= rhs` multiply coefficients and rhs by -1, or
    use the classmethod `from_greater_equal`.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    # def is_fulfilled(self, df_data: pd.DataFrame) -> bool:
    #     """Check if the linear inequality constraint is fulfilled in each row of the provided dataframe.

    #     Args:
    #         df_data (pd.DataFrame): Dataframe to evaluate constraint on.

    #     Returns:
    #         bool: True if fulfilled else False.
    #     """

    #     noise = 10e-10
    #     return (self.lhs(df_data) >= self.rhs - noise).all()

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        # noise = 10e-10 discuss with Behrang
        return self(experiments) <= 0

    def as_smaller_equal(self) -> Tuple[List[str], List[float], float]:
        """Return attributes in the smaller equal convention

        Returns:
            Tuple[List[str], List[float], float]: features, coefficients, rhs
        """
        return self.features, self.coefficients, self.rhs

    def as_greater_equal(self) -> Tuple[List[str], List[float], float]:
        """Return attributes in the greater equal convention

        Returns:
            Tuple[List[str], List[float], float]: features, coefficients, rhs
        """
        return self.features, [-1.0 * c for c in self.coefficients], -1.0 * self.rhs

    @classmethod
    def from_greater_equal(
        cls, features: List[str], coefficients: List[float], rhs: float
    ):
        """Class method to construct linear inequality constraint of the form `coefficients * x >= rhs`.

        Args:
            features (List[str]): List of feature keys.
            coefficients (List[float]): List of coefficients.
            rhs (float): Right-hand side of the constraint.
        """
        return cls(
            features=features,
            coefficients=[-1.0 * c for c in coefficients],
            rhs=-1.0 * rhs,
        )

    @classmethod
    def from_smaller_equal(
        cls, features: List[str], coefficients: List[float], rhs: float
    ):
        """Class method to construct linear inequality constraint of the form `coefficients * x <= rhs`.

        Args:
            features (List[str]): List of feature keys.
            coefficients (List[float]): List of coefficients.
            rhs (float): Right-hand side of the constraint.
        """
        return cls(
            features=features,
            coefficients=coefficients,
            rhs=rhs,
        )

    def __str__(self):
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        return super().__str__() + f" <= {self.rhs}"


class NonlinearConstraint(Constraint):
    expression: str

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return experiments.eval(self.expression)


class NonlinearEqualityConstraint(NonlinearConstraint):
    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(experiments), 0), index=experiments.index)

    def __str__(self):
        return f"{self.expression}==0"


class NonlinearInqualityConstraint(NonlinearConstraint):
    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        return self(experiments) <= 0

    def __str__(self):
        return f"{self.expression}<=0"


class NChooseKConstraint(Constraint):
    """NChooseK constraint that defines how many ingredients are allowed in a formulation.

    Attributes:
        features (List[str]): List of feature keys to which the constraint applies.
        min_count (int): Minimal number of non-zero/active feature values.
        max_count (int): Maximum number of non-zero/active feature values.
        none_also_valid (bool): In case that min_count > 0,
            this flag decides if zero active features are also allowed.
    """

    features: TFeatureKeys
    min_count: int
    max_count: int
    none_also_valid: bool

    @validator("features")
    def validate_features_unique(cls, features: List[str]):
        """Validates that provided feature keys are unique."""
        if len(features) != len(set(features)):
            raise ValueError("features must be unique")
        return features

    @root_validator(pre=False)
    def validate_counts(cls, values):
        """Validates if the minimum and maximum of allowed features are smaller than the overall number of features."""
        features = values["features"]
        min_count = values["min_count"]
        max_count = values["max_count"]

        if min_count > len(features):
            raise ValueError("min_count must be <= # of features")
        if max_count > len(features):
            raise ValueError("max_count must be <= # of features")
        if min_count > max_count:
            raise ValueError("min_values must be <= max_values")

        return values

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Check if the concurrency constraint is fulfilled for all the rows of the provided dataframe.

        Args:
            df_data (pd.DataFrame): Dataframe to evaluate constraint on.

        Returns:
            bool: True if fulfilled else False.
        """
        cols = self.features
        sums = (experiments[cols] > 0).sum(axis=1)

        lower = sums >= self.min_count
        upper = sums <= self.max_count

        if not self.none_also_valid:
            # return lower.all() and upper.all()
            return pd.Series(np.logical_and(lower, upper), index=experiments.index)
        else:
            none = sums == 0
            return pd.Series(
                np.logical_or(none, np.logical_and(lower, upper)),
                index=experiments.index,
            )

    def __str__(self):
        """Generate string representation of the constraint.

        Returns:
            str: string representation of the constraint.
        """
        res = (
            "of the features "
            + ", ".join(self.features)
            + f" between {self.min_count} and {self.max_count} must be used"
        )
        if self.none_also_valid:
            res += " (none is also ok)"
        return res


TConstraint = TypeVar("TConstraint", bound=Constraint)


class Constraints(BaseModel):

    constraints: List[Constraint] = Field(default_factory=lambda: [])

    def to_config(self) -> List:
        """Serializes a `Constraints` object.

        Returns:
            List: Constraints objects as serialized list.
        """
        return [constraint.to_config() for constraint in self.constraints]

    @classmethod
    def from_config(cls, config: List) -> "Constraints":
        """Instantiates a `Constraints` object based on the serialized list.

        Args:
            config (List): Serialized `Constraints` object as list.

        Returns:
            Constraints: Initialized `Constraints` object.
        """
        return cls(
            constraints=[Constraint.from_config(constraint) for constraint in config]
        )

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, i):
        return self.constraints[i]

    def __add__(self, other: "Constraints") -> "Constraints":
        return Constraints(constraints=self.constraints + other.constraints)

    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Numerically evaluate all constraints

        Args:
            experiments (pd.DataFrame): data to evaluate the constraint on

        Returns:
            pd.DataFrame: Constraint evaluation for each of the constraints
        """
        return pd.concat([c(experiments) for c in self.constraints], axis=1)

    def add(self, constraint: Constraint):
        """Add a new constraint to `self`.

        Args:
            constraint (Constraint): Constraint to add.
        """
        assert isinstance(constraint, Constraint)
        self.constraints.append(constraint)

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Check if all constraints are fulfilled on all rows of the provided dataframe

        Args:
            df_data (pd.DataFrame): Dataframe with data, the constraint validity should be tested on

        Returns:
            Boolean: True if all constraints are fulfilled for all rows, false if not
        """
        if len(self.constraints) == 0:
            return pd.Series([True] * len(experiments), index=experiments.index)
        return pd.concat(
            [c.is_fulfilled(experiments) for c in self.constraints], axis=1
        ).all(axis=1)

    def get(
        self,
        includes: Union[Type, List[Type]] = Constraint,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> "Constraints":
        """get constraints of the domain

        Args:
            includes (Union[Constraint, List[Constraint]], optional): Constraint class or list of specific constraint classes to be returned. Defaults to Constraint.
            excludes (Union[Type, List[Type]], optional): Constraint class or list of specific constraint classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[Constraint]: List of constraints in the domain fitting to the passed requirements.
        """
        return Constraints(
            constraints=filter_by_class(
                self.constraints,
                includes=includes,
                excludes=excludes,
                exact=exact,
            )
        )
