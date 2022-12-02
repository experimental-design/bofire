from typing import Optional

import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

from bofire.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective
from bofire.utils.study import Study


class SingleObjective(Study):
    def __init__(self, **data):
        super().__init__(**data)
        if (
            len(self.domain.output_features.get_by_objective(excludes=None)) > 1  # type: ignore
        ):  # TODO: update, when more features without DesFunc are implemented!
            raise ValueError("received multiobjective domain.")

    # TODO maybe unite with get_fbest from sobo, but not every strategy has get_fbest so far
    # and we have no universal way to compute it in domain --> maybe implement it also there.
    def get_fbest(self, experiments: Optional[pd.DataFrame] = None):
        if experiments is None:
            experiments = self.experiments
        ofeat = self.domain.output_features.get_by_objective(excludes=None)[0]  # type: ignore
        desirability = ofeat.desirability_function(experiments[ofeat.key])  # type: ignore
        return experiments.at[desirability.argmax(), ofeat.key]  # type: ignore


class Himmelblau(SingleObjective):

    use_constraints: bool = False
    best_possible_f: float = 0.0

    def setup_domain(self):
        domain = Domain()

        domain.add_feature(
            ContinuousInput(key="x_1", lower_bound=-4.0, upper_bound=4.0)
        )
        domain.add_feature(
            ContinuousInput(key="x_2", lower_bound=-4.0, upper_bound=6.0)
        )  # ToDo, check for correct bounds

        desirability_function = MinimizeObjective(w=1.0)
        domain.add_feature(
            ContinuousOutput(key="y", desirability_function=desirability_function)  # type: ignore
        )

        if self.use_constraints:
            raise ValueError("Not implemented yet!")
        return domain

    def run_candidate_experiments(self, candidates: pd.DataFrame, **kwargs):
        candidates.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
        candidates["valid_y"] = 1
        return candidates[self.domain.experiment_column_names].copy()  # type: ignore


class Ackley(SingleObjective):
    """Ackley function for testing optimization algorithms
    Virtual experiment corresponds to a function evaluation.
    Examples
    --------
    >>> b = Ackley()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)
    Notes
    -----
    This function is the negated version of https://en.wikipedia.org/wiki/Ackley_function.
    """

    num_categories: PositiveInt = 3
    categorical: bool = False
    descriptor: bool = False
    dim: PositiveInt = 2
    lower: float = -1
    upper: float = 3
    best_possible_f: float = 0.0
    evaluated_points = []

    # @validator("validate_categoricals")
    # def validate_categoricals(cls, v, num_categoricals):
    #     if v and num_categoricals ==1:
    #         raise ValueError("num_categories  must be specified if categorical=True")
    #     return v

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        if self.categorical:
            domain.add_feature(
                CategoricalInput(
                    key="category",
                    categories=[str(x) for x in range(self.num_categories)],
                )
            )

        if self.descriptor:
            domain.add_feature(
                CategoricalDescriptorInput(
                    key="descriptor",
                    categories=[str(x) for x in range(self.num_categories)],
                    descriptors=["d1"],
                    values=[[x * 2] for x in range(self.num_categories)],
                )
            )

        # continuous input features
        for d in range(self.dim):
            domain.add_feature(
                ContinuousInput(
                    key=f"x_{d+1}", lower_bound=self.lower, upper_bound=self.upper
                )
            )

        # Objective
        domain.add_feature(ContinuousOutput(key="y", objective=MaximizeObjective(w=1)))
        return domain

    def run_candidate_experiments(self, candidates, **kwargs):
        x = np.array([candidates[f"x_{d+1}"] for d in range(self.dim)])
        c = np.zeros(len(candidates))
        d = np.zeros(len(candidates))

        if self.categorical:
            # c = pd.to_numeric(candidates["category"], downcast="float")
            c = candidates.loc[:, "category"].values.astype(np.float64)
        if self.descriptor:
            d = candidates.loc[:, "descriptor"].values.astype(np.float64)

        z = x + c + d

        first_term = -20 * np.exp(-0.2 * np.sqrt(1 / self.dim * (z**2).sum()))
        second_term = -np.exp(1 / self.dim * (np.cos(2 * np.pi * z)).sum())
        y = -(first_term + second_term + 20 + np.exp(1) + (c + d) / 2)

        candidates["y"] = y
        candidates["valid_y"] = 1

        # save evaluated points for plotting
        self.evaluated_points.append(x.tolist())

        return candidates[self.domain.experiment_column_names].copy()  # type: ignore

    def reset(self):
        super().reset()
        self.evaluated_points = []
