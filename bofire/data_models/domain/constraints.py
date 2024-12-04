import collections.abc
from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Generic, List, Literal, Optional, Type, TypeVar, Union

import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import AnyConstraint, Constraint
from bofire.data_models.filters import filter_by_class


C = TypeVar("C", bound=Union[AnyConstraint, Constraint])
CIncludes = TypeVar("CIncludes", bound=Union[AnyConstraint, Constraint])
CExcludes = TypeVar("CExcludes", bound=Union[AnyConstraint, Constraint])


class Constraints(BaseModel, Generic[C]):
    type: Literal["Constraints"] = "Constraints"
    constraints: Sequence[C] = Field(default_factory=list)

    def __iter__(self) -> Iterator[C]:
        return iter(self.constraints)

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, i) -> C:
        return self.constraints[i]

    def __add__(
        self,
        other: Union[Sequence[CIncludes], "Constraints[CIncludes]"],
    ) -> "Constraints[Union[C, CIncludes]]":
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
                [c.is_fulfilled(experiments, tol) for c in self.constraints],
                axis=1,
            )
            .fillna(True)
            .all(axis=1)
        )

    def get(
        self,
        includes: Union[Type[CIncludes], Sequence[Type[CIncludes]]] = Constraint,
        excludes: Optional[Union[Type[CExcludes], List[Type[CExcludes]]]] = None,
        exact: bool = False,
    ) -> "Constraints[CIncludes]":
        """Get constraints of the domain

        Args:
            includes: Constraint class or list of specific constraint classes to be returned. Defaults to Constraint.
            excludes: Constraint class or list of specific constraint classes to be excluded from the return. Defaults to None.
            exact: Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            Constraints: constraints in the domain fitting to the passed requirements.

        """
        return Constraints(
            constraints=filter_by_class(
                self.constraints,
                includes=includes,
                excludes=excludes,
                exact=exact,
            ),
        )

    def get_reps_df(self):
        """Provides a tabular overwiev of all constraints within the domain

        Returns:
            pd.DataFrame: DataFrame listing all constraints of the domain with a description

        """
        df = pd.DataFrame(
            index=range(len(self.constraints)),
            columns=["Type", "Description"],
            data={
                "Type": [feat.__class__.__name__ for feat in self.get(Constraint)],
                "Description": [
                    constraint.__str__() for constraint in self.get(Constraint)
                ],
            },
        )
        return df
