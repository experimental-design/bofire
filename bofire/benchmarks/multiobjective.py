import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import validator
from pydantic.types import PositiveInt
from scipy.special import gamma

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain import Domain
from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    InputFeature,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objectives import MinimizeObjective
from bofire.utils.multiobjective import compute_hypervolume, get_pareto_front
from bofire.utils.study import Study


class MultiObjective(Study):

    ref_point: Optional[dict]

    def get_fbest(self, experiments: Optional[pd.DataFrame] = None):
        if experiments is None:
            experiments = self.experiments
        optimal_experiments = get_pareto_front(self.domain, experiments)  # type: ignore
        return compute_hypervolume(self.domain, optimal_experiments, self.ref_point)  # type: ignore

    def __init__(self, **data):
        super().__init__(**data)
        if (
            len(self.domain.output_features.get_by_objective(excludes=None)) < 2  # type: ignore
        ):  # TODO: update, when more features without DesFunc are implemented!
            raise ValueError("received singelobjective domain.")


class DTLZ2(Benchmark):
    """Multiobjective bechmark function for testing optimization algorithms.
    Info about the function: https://pymoo.org/problems/many/dtlz.html
    """

    def __init__(
        self, dim: PositiveInt, k: Optional[int], num_objectives: PositiveInt = 2
    ):
        self.num_objectives = num_objectives
        self.dim = dim
        self.k = k

        input_features = []
        for i in range(self.dim):
            input_features.append(
                ContinuousInput(key="x_%i" % (i), lower_bound=0.0, upper_bound=1.0)
            )
        output_features = []
        self.k = self.dim - self.num_objectives + 1
        for i in range(self.num_objectives):
            output_features.append(
                ContinuousOutput(key=f"f_{i}", objective=MinimizeObjective(w=1.0))
            )
        domain = Domain(
            input_features=InputFeatures(features=input_features),
            output_features=OutputFeatures(features=output_features),
        )
        self.ref_point = {
            feat: 1.1 for feat in domain.get_feature_keys(ContinuousOutput)
        }
        self._domain = domain

    @validator("dim")
    def validate_dim(cls, dim, values):
        num_objectives = values["num_objectives"]
        if dim <= values["num_objectives"]:
            raise ValueError(
                f"dim must be > num_objectives, but got {dim} and {num_objectives}."
            )
        return dim

    @property
    def best_possible_hypervolume(self) -> float:
        # hypercube - volume of hypersphere in R^d such that all coordinates are
        # positive
        hypercube_vol = self.ref_point[0] ** self.num_objectives  # type: ignore
        pos_hypersphere_vol = (
            math.pi ** (self.num_objectives / 2)
            / gamma(self.num_objectives / 2 + 1)
            / 2**self.num_objectives
        )
        return hypercube_vol - pos_hypersphere_vol

    def f(self, candidates):
        X = candidates[self.domain.get_feature_keys(InputFeature)].values  # type: ignore
        X_m = X[..., -self.k :]  # type: ignore
        g_X = ((X_m - 0.5) ** 2).sum(axis=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.copy()
            f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
            if i > 0:
                f_i *= np.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        candidates[
            self.domain.output_features.get_keys_by_objective(excludes=None)  # type: ignore
        ] = np.stack(fs, axis=-1)
        candidates[
            [
                "valid_%s" % feat
                for feat in self.domain.output_features.get_keys_by_objective(  # type: ignore
                    excludes=None
                )
            ]
        ] = 1
        return candidates[self.domain.experiment_column_names].copy()  # type: ignore
