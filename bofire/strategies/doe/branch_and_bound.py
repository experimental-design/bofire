from __future__ import annotations

from functools import total_ordering
from queue import PriorityQueue

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
        self, fixed_experiments: pd.DataFrame, design_matrix: pd.DataFrame, value: float
    ):
        self.fixed_experiments = fixed_experiments
        self.design_matrix = design_matrix
        self.value = value

    def get_next_fixed_experiments(self):
        b1 = 1
        b2 = 2
        return b1, b2

    def __eq__(self, other: NodeExperiment):
        return self.value == other.value

    def __ne__(self, other: NodeExperiment):
        return self.value != other.value

    def __lt__(self, other: NodeExperiment):
        return self.value < other.value


def is_valid(design_matrix: pd.DataFrame, domain: Domain):
    categorical_vars = domain.get_features(includes=ContinuousBinaryInput)
    for var in categorical_vars:
        value = design_matrix.get(var.key)
        if not (np.isclose(value, 0) or np.isclose(value, 1)):
            return False

    return True


def bnb(priority_queue: PriorityQueue, **kwargs):
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
    current_design = current_branch.design_matrix
    if is_valid(current_design, domain):
        return current_branch

    # branch current solutions in sub-problems
    next_branches = current_branch.get_next_fixed_experiments()

    # solve branched problems
    for branch in next_branches:
        solution = find_local_max_ipopt(**kwargs)
        value = objective_class.evaluate(solution.to_numpy().T)
        new_node = NodeExperiment(branch, solution, value)
        priority_queue.put(new_node)

    bnb(priority_queue, **kwargs)
