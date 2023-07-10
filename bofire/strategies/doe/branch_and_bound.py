from __future__ import annotations

from functools import total_ordering
from queue import PriorityQueue
from typing import List

import numpy as np
import pandas as pd

from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import ContinuousBinaryInput
from bofire.strategies.doe.design import find_local_max_ipopt
from bofire.strategies.doe.objective import get_objective_class
from bofire.strategies.doe.utils import get_formula_from_string
from bofire.strategies.enum import OptimalityCriterionEnum


@total_ordering
class NodeExperiment:
    def __init__(
        self,
        fixed_experiments: pd.DataFrame,
        design_matrix: pd.DataFrame,
        value: float,
        categorical_groups: List[List[ContinuousBinaryInput]],
    ):
        self.fixed_experiments = fixed_experiments
        self.design_matrix = design_matrix
        self.value = value
        self.categorical_groups = categorical_groups

    def get_next_fixed_experiments(self) -> List[pd.DataFrame]:

        for group in self.categorical_groups:
            for row_index, _exp in self.fixed_experiments.iterrows():
                if self.fixed_experiments.iloc[row_index][group[0].key] is None:
                    current_keys = [elem.key for elem in group]
                    allowed_fixations = np.eye(len(group))
                    branches = [
                        self.fixed_experiments.copy()
                        for i in range(len(allowed_fixations))
                    ]
                    for k, elem in enumerate(branches):
                        elem.loc[row_index, current_keys] = allowed_fixations[k]
                    return branches
        return []

    def __eq__(self, other: NodeExperiment) -> bool:
        return self.value == other.value

    def __ne__(self, other: NodeExperiment) -> bool:
        return self.value != other.value

    def __lt__(self, other: NodeExperiment) -> bool:
        return self.value < other.value

    def __str__(self):
        return (
            "\n ================ Branch-and-Bound Node ================ \n"
            + f"objective value: {self.value} \n"
            + f"design matrix: \n{self.design_matrix.round(4)} \n"
            + f"current fixations: \n{self.fixed_experiments.round(4)} \n"
        )


def is_valid(design_matrix: pd.DataFrame, domain: Domain) -> bool:
    categorical_vars = domain.get_features(includes=ContinuousBinaryInput)
    for var in categorical_vars:
        value = design_matrix.get(var.key)
        if not (np.logical_or(np.isclose(value, 0), np.isclose(value, 1)).all()):
            return False

    return True


def bnb(priority_queue: PriorityQueue, **kwargs) -> NodeExperiment:
    if priority_queue.empty():
        raise RuntimeError("Queue empty before feasible solution was found")

    domain = kwargs["domain"]
    n_experiments = kwargs["n_experiments"]

    model_formula = get_formula_from_string(
        model_type="linear", rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(OptimalityCriterionEnum.D_OPTIMALITY)
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments
    )

    current_branch = priority_queue.get()
    # test if current solution is already valid
    if is_valid(current_branch.design_matrix, domain):
        return current_branch

    # branch current solutions in sub-problems
    next_branches = current_branch.get_next_fixed_experiments()

    # solve branched problems
    for _i, branch in enumerate(next_branches):
        design = find_local_max_ipopt(partially_fixed_experiments=branch, **kwargs)
        value = objective_class.evaluate(design.to_numpy().flatten())
        new_node = NodeExperiment(
            branch, design, value, current_branch.categorical_groups
        )
        priority_queue.put(new_node)

    return bnb(priority_queue, **kwargs)
