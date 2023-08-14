from __future__ import annotations

from functools import total_ordering
from queue import PriorityQueue
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import ConstraintNotFulfilledError
from bofire.data_models.features.api import ContinuousInput
from bofire.strategies.doe.design import find_local_max_ipopt
from bofire.strategies.doe.objective import get_objective_class
from bofire.strategies.doe.utils import get_formula_from_string
from bofire.strategies.doe.utils_categorical_discrete import equal_count_split


@total_ordering
class NodeExperiment:
    def __init__(
        self,
        partially_fixed_experiments: pd.DataFrame,
        design_matrix: pd.DataFrame,
        value: float,
        categorical_groups: Optional[List[List[ContinuousInput]]] = None,
        discrete_vars: Optional[Dict[str, Tuple[ContinuousInput, List[float]]]] = None,
    ):
        """

        Args:
            partially_fixed_experiments: dataframe containing (some) fixed variables for experiments.
            design_matrix: optimal design for given the fixed and partially fixed experiments
            value: value of the objective function evaluated with the design_matrix
            categorical_groups: Represents the different groups of the categorical variables
            discrete_vars: Dict of discrete variables and the corresponding valid values in the optimization problem
        """
        self.partially_fixed_experiments = partially_fixed_experiments
        self.design_matrix = design_matrix
        self.value = value
        if categorical_groups is not None:
            self.categorical_groups = categorical_groups
        else:
            self.categorical_groups = []
        if discrete_vars is not None:
            self.discrete_vars = discrete_vars
        else:
            self.discrete_vars = {}

    def get_next_fixed_experiments(self) -> List[pd.DataFrame]:
        """
        Based on the current partially_fixed_experiment DataFrame the next branches are determined. One variable will
        be fixed more than before.
        Returns: List of the next possible branches where only one variable more is fixed

        """
        # branching for the binary/ categorical variables
        for group in self.categorical_groups:
            for row_index, _exp in self.partially_fixed_experiments.iterrows():
                if (
                    self.partially_fixed_experiments.iloc[row_index][group[0].key]
                    is None
                ):
                    current_keys = [elem.key for elem in group]
                    allowed_fixations = np.eye(len(group))
                    branches = [
                        self.partially_fixed_experiments.copy()
                        for i in range(len(allowed_fixations))
                    ]
                    for k, elem in enumerate(branches):
                        elem.loc[row_index, current_keys] = allowed_fixations[k]
                    return branches

        # branching for the discrete variables
        for key, (var, values) in self.discrete_vars.items():
            for row_index, _exp in self.partially_fixed_experiments.iterrows():
                current_fixation = self.partially_fixed_experiments.iloc[row_index][key]
                first_fixation, second_fixation = None, None
                if current_fixation is None:
                    lower_split, upper_split = equal_count_split(
                        values, var.lower_bound, var.upper_bound
                    )
                    first_fixation = (var.lower_bound, lower_split)
                    second_fixation = (upper_split, var.upper_bound)

                elif current_fixation[0] != current_fixation[1]:
                    lower_split, upper_split = equal_count_split(
                        values, current_fixation[0], current_fixation[1]
                    )
                    first_fixation = (current_fixation[0], lower_split)
                    second_fixation = (upper_split, current_fixation[1])

                if first_fixation is not None:
                    first_branch = self.partially_fixed_experiments.copy()
                    second_branch = self.partially_fixed_experiments.copy()

                    first_branch.loc[row_index, key] = first_fixation
                    second_branch.loc[row_index, key] = second_fixation

                    return [first_branch, second_branch]

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
            + f"current fixations: \n{self.partially_fixed_experiments.round(4)} \n"
        )


def is_valid(node: NodeExperiment, tolerance: float = 1e-2) -> bool:
    """
    test if a design is a valid solution. i.e. binary and discrete variables are valid
    Args:
        node: the current node of the branch to be tested
        tolerance: absolute tolerance between valid values and values in the design

    Returns: True if the design is valid, else False

    """
    categorical_vars = [var for group in node.categorical_groups for var in group]
    design_matrix = node.design_matrix
    for var in categorical_vars:
        value = design_matrix.get(var.key)
        if not (
            np.logical_or(
                np.isclose(value, 0, atol=tolerance),
                np.isclose(value, 1, atol=tolerance),
            ).all()
        ):
            return False

    discrete_vars = node.discrete_vars
    for _key, (var, values) in discrete_vars.items():
        value = design_matrix.get(var.key)
        if False in [True in np.isclose(v, values, atol=tolerance) for v in value]:  # type: ignore
            return False
    return True


def bnb(
    priority_queue: PriorityQueue,
    verbose: bool = False,
    num_explored: int = 0,
    **kwargs,
) -> NodeExperiment:
    """
    branch-and-bound algorithm for solving optimization problems containing binary and discrete variables
    Args:
        num_explored: keeping track of how many branches have been explored
        priority_queue (PriorityQueue): initial nodes of the branching tree
        verbose (bool): if true, print information during the optimization process
        **kwargs: parameters for the actual optimization / find_local_max_ipopt

    Returns: a branching Node containing the best design found

    """
    if priority_queue.empty():
        raise RuntimeError("Queue empty before feasible solution was found")

    domain = kwargs["domain"]
    n_experiments = kwargs["n_experiments"]

    # get objective function
    model_formula = get_formula_from_string(
        model_type=kwargs["model_type"], rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(kwargs["objective"])
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments
    )

    pre_size = priority_queue.qsize()
    current_branch = priority_queue.get()
    # test if current solution is already valid
    if is_valid(current_branch):
        return current_branch

    # branch current solutions in sub-problems
    next_branches = current_branch.get_next_fixed_experiments()

    if verbose:
        print(
            f"current length of branching queue (+ new branches): {pre_size} + {len(next_branches)} currently "
            f"explored branches: {num_explored}, current best value: {current_branch.value}"
        )
    # solve branched problems
    for _i, branch in enumerate(next_branches):
        kwargs["sampling"] = current_branch.design_matrix
        try:
            design = find_local_max_ipopt(partially_fixed_experiments=branch, **kwargs)
            value = objective_class.evaluate(design.to_numpy().flatten())
            new_node = NodeExperiment(
                branch,
                design,
                value,
                current_branch.categorical_groups,
                current_branch.discrete_vars,
            )
            domain.validate_candidates(
                candidates=design.apply(lambda x: np.round(x, 8)),
                only_inputs=True,
                tol=1e-4,
                raise_validation_error=True,
            )

            priority_queue.put(new_node)
        except ConstraintNotFulfilledError:
            if verbose:
                print("skipping branch because of not fulfilling constraints")

    return bnb(
        priority_queue,
        verbose=verbose,
        num_explored=num_explored + len(next_branches),
        **kwargs,
    )
