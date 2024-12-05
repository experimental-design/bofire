from __future__ import annotations

import time
from functools import total_ordering
from itertools import combinations_with_replacement, product
from queue import PriorityQueue
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import ConstraintNotFulfilledError
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, Input
from bofire.data_models.strategies.doe import AnyOptimalityCriterion
from bofire.strategies.doe.design import find_local_max_ipopt
from bofire.strategies.doe.objective import get_objective_function
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
        """Args:
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
        """Based on the current partially_fixed_experiment DataFrame the next branches are determined. One variable will
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
                        values,
                        var.lower_bound,
                        var.upper_bound,
                    )
                    first_fixation = (var.lower_bound, lower_split)
                    second_fixation = (upper_split, var.upper_bound)

                elif current_fixation[0] != current_fixation[1]:
                    lower_split, upper_split = equal_count_split(
                        values,
                        current_fixation[0],
                        current_fixation[1],
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
    """Test if a design is a valid solution. i.e. binary and discrete variables are valid
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
        if False in [True in np.isclose(v, values, atol=tolerance) for v in value]:
            return False
    return True


def bnb(
    priority_queue: PriorityQueue,
    verbose: bool = False,
    num_explored: int = 0,
    **kwargs,
) -> NodeExperiment:
    """branch-and-bound algorithm for solving optimization problems containing binary and discrete variables
    Args:
        num_explored: keeping track of how many branches have been explored
        priority_queue (PriorityQueue): initial nodes of the branching tree
        verbose (bool): if true, print information during the optimization process
        **kwargs: parameters for the actual optimization / find_local_max_ipopt

    Returns: a branching Node containing the best design found

    """
    if priority_queue.empty():
        raise RuntimeError("Queue empty before feasible solution was found")

    objective_function = get_objective_function(
        criterion=kwargs["criterion"],
        domain=kwargs["domain"],
        n_experiments=kwargs["n_experiments"],
    )
    assert objective_function is not None, "Criterion type is not supported!"

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
            f"explored branches: {num_explored}, current best value: {current_branch.value}",
        )
    # solve branched problems
    for _i, branch in enumerate(next_branches):
        kwargs["sampling"] = current_branch.design_matrix
        try:
            design = find_local_max_ipopt(partially_fixed_experiments=branch, **kwargs)
            value = objective_function.evaluate(design.to_numpy().flatten())
            new_node = NodeExperiment(
                branch,
                design,
                value,
                current_branch.categorical_groups,
                current_branch.discrete_vars,
            )
            kwargs["domain"].validate_candidates(
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


def find_local_max_ipopt_BaB(
    domain: Domain,
    n_experiments: int,
    criterion: AnyOptimalityCriterion,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    partially_fixed_experiments: Optional[pd.DataFrame] = None,
    categorical_groups: Optional[List[List[ContinuousInput]]] = None,
    discrete_variables: Optional[Dict[str, Tuple[ContinuousInput, List[float]]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given domain and model.
    It allows for the problem to have categorical values which is solved by Branch-and-Bound
        Args:
            domain (Domain): domain containing the inputs and constraints.
            model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
                are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
            n_experiments (int): Number of experiments. By default the value corresponds to
                the number of model terms - dimension of ker() + 3.
            delta (float): Regularization parameter. Default value is 1e-3.
            ipopt_options (Dict, optional): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
            sampling (pd.DataFrame): dataframe containing the initial guess.
            fixed_experiments (pd.DataFrame): dataframe containing experiments that will be definitely part of the design.
                Values are set before the optimization.
            partially_fixed_experiments (pd.DataFrame): dataframe containing (some) fixed variables for experiments.
                Values are set before the optimization. Within one experiment not all variables need to be fixed.
                Variables can be fixed to one value or can be set to a range by setting a tuple with lower and upper bound
                Non-fixed variables have to be set to None or nan.
            objective (OptimalityCriterionEnum): OptimalityCriterionEnum object indicating which objective function to use.
            categorical_groups (Optional[List[List[ContinuousInput]]]). Represents the different groups of the
               relaxed categorical variables. Defaults to None.
            discrete_variables (Optional[Dict[str, Tuple[ContinuousInput, List[float]]]]): dict of relaxed discrete inputs
                with key:(relaxed variable, valid values). Defaults to None
            verbose (bool): if true, print information during the optimization process
            transform_range (Optional[Bounds]): range to which the input variables are transformed.
                If None is provided, the features will not be scaled. Defaults to None.
        Returns:
            A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
            local optimum.
    """
    from bofire.strategies.doe.branch_and_bound import NodeExperiment, bnb

    if categorical_groups is None:
        categorical_groups = []

    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )
    assert objective_function is not None, "Criterion type is not supported!"

    # setting up initial node in the branch-and-bound tree
    column_keys = domain.inputs.get_keys()

    if fixed_experiments is not None:
        subtract = len(fixed_experiments)
        initial_branch = pd.DataFrame(
            np.full((n_experiments - subtract, len(column_keys)), None),
            columns=column_keys,
        )
        initial_branch = pd.concat([fixed_experiments, initial_branch]).reset_index(
            drop=True
        )
    else:
        initial_branch = pd.DataFrame(
            np.full((n_experiments, len(column_keys)), None),
            columns=column_keys,
        )

    if partially_fixed_experiments is not None:
        partially_fixed_experiments = pd.concat(
            [
                partially_fixed_experiments,
                pd.DataFrame(
                    np.full(
                        (
                            n_experiments - len(partially_fixed_experiments),
                            len(domain.inputs),
                        ),
                        None,
                    ),
                    columns=domain.inputs.get_keys(includes=Input),
                ),
            ]
        ).reset_index(drop=True)

        initial_branch.mask(
            partially_fixed_experiments.notnull(),  # type: ignore
            other=partially_fixed_experiments,
            inplace=True,
        )

    initial_design = find_local_max_ipopt(
        domain,
        n_experiments,
        ipopt_options,
        sampling,
        None,
        partially_fixed_experiments=initial_branch,
        criterion=criterion,
    )
    initial_value = objective_function.evaluate(
        initial_design.to_numpy().flatten(),
    )

    initial_node = NodeExperiment(
        initial_branch,
        initial_design,
        initial_value,
        categorical_groups,
        discrete_variables,
    )

    # initializing branch-and-bound queue
    initial_queue = PriorityQueue()
    initial_queue.put(initial_node)

    # starting branch-and-bound
    result_node = bnb(
        initial_queue,
        domain=domain,
        n_experiments=n_experiments,
        ipopt_options=ipopt_options,
        sampling=sampling,
        fixed_experiments=None,
        criterion=criterion,
        verbose=verbose,
    )

    return result_node.design_matrix


def find_local_max_ipopt_exhaustive(
    domain: Domain,
    n_experiments: int,
    criterion: AnyOptimalityCriterion,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    partially_fixed_experiments: Optional[pd.DataFrame] = None,
    categorical_groups: Optional[List[List[ContinuousInput]]] = None,
    discrete_variables: Optional[Dict[str, Tuple[ContinuousInput, List[float]]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given domain and model.
    It allows for the problem to have categorical values which is solved by exhaustive search
        Args:
            domain (Domain): domain containing the inputs and constraints.
            model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
                are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
            n_experiments (int): Number of experiments. By default the value corresponds to
                the number of model terms - dimension of ker() + 3.
            delta (float): Regularization parameter. Default value is 1e-3.
            ipopt_options (Dict, optional): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
            sampling (pd.DataFrame): dataframe containing the initial guess.
            fixed_experiments (pd.DataFrame): dataframe containing experiments that will be definitely part of the design.
                Values are set before the optimization.
            objective (OptimalityCriterionEnum): OptimalityCriterionEnum object indicating which objective function to use.
            partially_fixed_experiments (pd.DataFrame): dataframe containing (some) fixed variables for experiments.
                Values are set before the optimization. Within one experiment not all variables need to be fixed.
                Variables can be fixed to one value or can be set to a range by setting a tuple with lower and upper bound
                Non-fixed variables have to be set to None or nan.
            categorical_groups (Optional[List[List[ContinuousInput]]]). Represents the different groups of the
               relaxed categorical variables. Defaults to None.
            discrete_variables (Optional[Dict[str, Tuple[ContinuousInput, List[float]]]]): dict of relaxed discrete inputs
                with key:(relaxed variable, valid values). Defaults to None
            verbose (bool): if true, print information during the optimization process
            transform_range (Optional[Bounds]): range to which the input variables are transformed.
        Returns:
            A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
            local optimum.
    """

    if categorical_groups is None:
        categorical_groups = []

    if discrete_variables is not None or len(discrete_variables) > 0:  # type: ignore
        raise NotImplementedError(
            "Exhaustive search for discrete variables is not implemented yet."
        )

    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )
    assert objective_function is not None, "Criterion type is not supported!"

    # get binary variables
    binary_vars = [var for group in categorical_groups for var in group]
    list_keys = [var.key for var in binary_vars]

    # determine possible fixations of the different categories
    allowed_fixations = []
    for group in categorical_groups:
        allowed_fixations.append(np.eye(len(group)))

    n_non_fixed_experiments = n_experiments
    if fixed_experiments is not None:
        n_non_fixed_experiments -= len(fixed_experiments)

    allowed_fixations = product(*allowed_fixations)
    all_n_fixed_experiments = combinations_with_replacement(
        allowed_fixations, n_non_fixed_experiments
    )

    if partially_fixed_experiments is not None:
        partially_fixed_experiments = pd.concat(
            [
                partially_fixed_experiments,
                pd.DataFrame(
                    np.full(
                        (
                            n_non_fixed_experiments - len(partially_fixed_experiments),
                            len(domain.inputs),
                        ),
                        None,
                    ),
                    columns=domain.inputs.get_keys(includes=Input),
                ),
            ]
        ).reset_index(drop=True)

    # testing all different fixations
    column_keys = domain.inputs.get_keys()
    group_keys = [var.key for group in categorical_groups for var in group]
    minimum = float("inf")
    optimal_design = pd.DataFrame()
    all_n_fixed_experiments = list(all_n_fixed_experiments)
    for i, binary_fixed_experiments in enumerate(all_n_fixed_experiments):
        if verbose:
            start_time = time.time()
        # setting up the pd.Dataframe for the partially fixed experiment
        binary_fixed_experiments = np.array(
            [
                var
                for experiment in binary_fixed_experiments
                for group in experiment
                for var in group
            ]
        ).reshape(n_non_fixed_experiments, len(binary_vars))

        binary_fixed_experiments = pd.DataFrame(
            binary_fixed_experiments, columns=group_keys
        )
        one_set_of_experiments = pd.DataFrame(
            np.full((n_non_fixed_experiments, len(domain.inputs)), None),
            columns=column_keys,
        )

        one_set_of_experiments.mask(
            binary_fixed_experiments.notnull(),
            other=binary_fixed_experiments,
            inplace=True,
        )

        if partially_fixed_experiments is not None:
            one_set_of_experiments.mask(
                partially_fixed_experiments.notnull(),
                other=partially_fixed_experiments,
                inplace=True,
            )

        if fixed_experiments is not None:
            one_set_of_experiments = pd.concat(
                [fixed_experiments, one_set_of_experiments]
            ).reset_index(drop=True)

        if sampling is not None:
            sampling.loc[:, list_keys] = one_set_of_experiments[list_keys].to_numpy()

        # minimizing with the current fixation
        try:
            current_design = find_local_max_ipopt(
                domain,
                n_experiments,
                ipopt_options,
                sampling,
                None,
                one_set_of_experiments,
                criterion=criterion,
            )
            domain.validate_candidates(
                candidates=current_design.apply(lambda x: np.round(x, 8)),
                only_inputs=True,
                tol=1e-4,
                raise_validation_error=True,
            )
            temp_value = objective_function.evaluate(
                current_design.to_numpy().flatten(),
            )
            if minimum is None or minimum > temp_value:
                minimum = temp_value
                optimal_design = current_design
            if verbose:
                print(
                    f"branch: {i} / {len(all_n_fixed_experiments)}, "
                    f"time: {time.time() - start_time},"  # type: ignore
                    f"solution: {temp_value}, minimum after run {minimum},"
                    f"difference: {temp_value - minimum}"
                )
        except ConstraintNotFulfilledError:
            if verbose:
                print("skipping branch because of not fulfilling constraints")
    return optimal_design
