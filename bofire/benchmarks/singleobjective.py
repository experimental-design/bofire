from typing import Optional

import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objectives import MaximizeObjective
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


class Ackley(Benchmark):
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

    # @validator("validate_categoricals")
    # def validate_categoricals(cls, v, num_categoricals):
    #     if v and num_categoricals ==1:
    #         raise ValueError("num_categories  must be specified if categorical=True")
    #     return v

    def __init__(
        self,
        num_categories: PositiveInt = 3,
        categorical: bool = False,
        descriptor: bool = False,
        dim: PositiveInt = 2,
        lower: float = -32.768,
        upper: float = 32.768,
        best_possible_f: float = 0.0,
        evaluated_points=[],
    ):

        self.num_categories = num_categories
        self.categorical = categorical
        self.descriptor = descriptor
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.best_possible_f = best_possible_f
        self.evaluated_points = evaluated_points

        input_feature_list = []
        # Decision variables
        if self.categorical:
            input_feature_list.append(
                CategoricalInput(
                    key="category",
                    categories=[str(x) for x in range(self.num_categories)],
                )
            )

        if self.descriptor:
            input_feature_list.append(
                CategoricalDescriptorInput(
                    key="descriptor",
                    categories=[str(x) for x in range(self.num_categories)],
                    descriptors=["d1"],
                    values=[[x * 2] for x in range(self.num_categories)],
                )
            )

        # continuous input features
        for d in range(self.dim):
            input_feature_list.append(
                ContinuousInput(
                    key=f"x_{d+1}", lower_bound=self.lower, upper_bound=self.upper
                )
            )

        # Objective
        output_feature = ContinuousOutput(key="y", objective=MaximizeObjective(w=1))

        self._domain = Domain(
            input_features=InputFeatures(features=input_feature_list),
            output_features=OutputFeatures(features=[output_feature]),
        )

    def f(self, X, **kwargs):
        a = 20
        b = 0.2
        c = np.pi * 2
        x = np.array([X[f"x_{d+1}"] for d in range(self.dim)])

        c = np.zeros(len(X))
        d = np.zeros(len(X))
        n = self.dim

        if self.categorical:
            # c = pd.to_numeric(X["category"], downcast="float")
            c = X.loc[:, "category"].values.astype(np.float64)
        if self.descriptor:
            d = X.loc[:, "descriptor"].values.astype(np.float64)

        z = x + c + d

        term1 = -a * np.exp(-b * ((1 / n) * np.sum(z**2, axis=0)) ** 0.5)
        term2 = -np.exp((1 / n) * np.sum(np.cos(c * z), axis=0))
        term3 = a + np.exp(1)
        y = term1 + term2 + term3

        X["y"] = y
        X["valid_y"] = 1

        # save evaluated points for plotting
        self.evaluated_points.append(x.tolist())
        return X[self.domain.experiment_column_names].copy()  # type: ignore

    def get_optima(self) -> pd.DataFrame:
        x = np.zeros((1, self.dim))
        y = 0
        return pd.DataFrame(
            np.c_[x, y],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )
