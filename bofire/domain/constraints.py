import collections.abc
from itertools import chain
from typing import List, Literal, Sequence, Type, Union

import pandas as pd
from pydantic import Field

from bofire.any.constraint import AnyConstraint
from bofire.domain.constraint import Constraint
from bofire.domain.util import PydanticBaseModel, filter_by_class


class Constraints(PydanticBaseModel):

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
