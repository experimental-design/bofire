import collections.abc
from itertools import chain
from typing import List, Literal, Sequence, Type, Union

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel, filter_by_class
from bofire.data_models.constraints.api import AnyConstraint, Constraint


class Constraints(BaseModel):
    type: Literal["Constraints"] = "Constraints"
    constraints: Sequence[AnyConstraint] = Field(default_factory=lambda: [])

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, i):
        return self.constraints[i]

    def __add__(
        self, other: Union[Sequence[AnyConstraint], "Constraints"]
    ) -> "Constraints":
        if isinstance(other, collections.abc.Sequence):
            other_constraints = other
        else:
            other_constraints = other.constraints
        constraints = list(chain(self.constraints, other_constraints))
        return Constraints(constraints=constraints)

    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Numerically evaluate all constraints

        Args:
            experiments (pd.DataFrame): data to evaluate the constraint on

        Returns:
            pd.DataFrame: Constraint evaluation for each of the constraints
        """
        return pd.concat([c(experiments) for c in self.constraints], axis=1)

    def jacobian(self, experiments: pd.DataFrame) -> list:
        """Numerically evaluate the jacobians of all constraints

        Args:
            experiments (pd.DataFrame): data to evaluate the constraint jacobians on

        Returns:
            list: A list containing the jacobians as pd.DataFrames
        """
        return [c.jacobian(experiments) for c in self.constraints]

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        """Check if all constraints are fulfilled on all rows of the provided dataframe

        Args:
            experiments (pd.DataFrame): Dataframe with data, the constraint validity should be tested on
            tol (float, optional): tolerance parameter. A constraint is considered as not fulfilled if
                the violation is larger than tol. Defaults to 0.

        Returns:
            Boolean: True if all constraints are fulfilled for all rows, false if not
        """
        if len(self.constraints) == 0:
            return pd.Series([True] * len(experiments), index=experiments.index)
        return (
            pd.concat(
                [c.is_fulfilled(experiments, tol) for c in self.constraints], axis=1
            )
            .fillna(True)
            .all(axis=1)
        )

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
