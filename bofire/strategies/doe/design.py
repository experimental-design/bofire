import time
import warnings
from itertools import combinations_with_replacement, product
from queue import PriorityQueue
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.optimize._minimize import standardize_constraints

from bofire.data_models.constraints.api import (
    ConstraintNotFulfilledError,
    NChooseKConstraint,
    NonlinearConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput, Input
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.doe.objective import get_objective_class
from bofire.strategies.doe.utils import (
    constraints_as_scipy_constraints,
    get_formula_from_string,
    metrics,
    nchoosek_constraints_as_bounds,
)
from bofire.strategies.enum import OptimalityCriterionEnum
from bofire.strategies.samplers.polytope import PolytopeSampler


def find_local_max_ipopt_BaB(
    domain: Domain,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    delta: float = 1e-7,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    partially_fixed_experiments: Optional[pd.DataFrame] = None,
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
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
        Returns:
            A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
            local optimum.
    """
    from bofire.strategies.doe.branch_and_bound import NodeExperiment, bnb

    if categorical_groups is None:
        categorical_groups = []

    n_experiments = get_n_experiments(
        domain=domain, model_type=model_type, n_experiments=n_experiments
    )

    # get objective function
    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(objective)
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments, delta=delta
    )

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
                    columns=domain.get_feature_keys(includes=Input),
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
        model_type,
        n_experiments,
        delta,
        ipopt_options,
        sampling,
        None,
        partially_fixed_experiments=initial_branch,
        objective=objective,
    )
    initial_value = objective_class.evaluate(
        initial_design.to_numpy().flatten(),
    )

    initial_node = NodeExperiment(
        initial_branch,
        initial_design,
        initial_value,
        categorical_groups,
        discrete_variables,  # type: ignore
    )

    # initializing branch-and-bound queue
    initial_queue = PriorityQueue()
    initial_queue.put(initial_node)

    # starting branch-and-bound
    result_node = bnb(
        initial_queue,
        domain=domain,
        model_type=model_type,
        n_experiments=n_experiments,
        delta=delta,
        ipopt_options=ipopt_options,
        sampling=sampling,
        fixed_experiments=None,
        objective=objective,
        verbose=verbose,
    )

    return result_node.design_matrix


def find_local_max_ipopt_exhaustive(
    domain: Domain,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    delta: float = 1e-7,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
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

    # get objective function
    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(objective)
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments, delta=delta
    )

    # get binary variables
    binary_vars = [var for group in categorical_groups for var in group]
    list_keys = [var.key for var in binary_vars]

    # determine possible fixations of the different categories
    allowed_fixations = []
    for group in categorical_groups:
        allowed_fixations.append(np.eye(len(group)))

    n_experiments = get_n_experiments(
        domain=domain, model_type=model_type, n_experiments=n_experiments
    )
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
                    columns=domain.get_feature_keys(includes=Input),
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
                model_type,
                n_experiments,
                delta,
                ipopt_options,
                sampling,
                None,
                one_set_of_experiments,
                objective,
            )
            domain.validate_candidates(
                candidates=current_design.apply(lambda x: np.round(x, 8)),
                only_inputs=True,
                tol=1e-4,
                raise_validation_error=True,
            )
            temp_value = objective_class.evaluate(
                current_design.to_numpy().flatten(),
            )
            if minimum is None or minimum > temp_value:
                minimum = temp_value
                optimal_design = current_design
            if verbose:
                print(
                    f"branch: {i} / {len(all_n_fixed_experiments)}, time: {time.time() - start_time} solution: {temp_value}, minimum after run {minimum}, difference: {temp_value - minimum}"  # type: ignore
                )
        except ConstraintNotFulfilledError:
            if verbose:
                print("skipping branch because of not fulfilling constraints")
    return optimal_design


def find_local_max_ipopt(
    domain: Domain,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    delta: float = 1e-7,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    partially_fixed_experiments: Optional[pd.DataFrame] = None,
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
) -> pd.DataFrame:
    """Function computing an optimal design for a given domain and model.
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
    Returns:
        A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
        local optimum.
    """

    #
    # Checks and preparation steps
    #

    # warn user if IPOPT scipy interface is not available
    try:
        from cyipopt import minimize_ipopt  # type: ignore
    except ImportError as e:
        warnings.warn(e.msg)
        warnings.warn(
            "please run `conda install -c conda-forge cyipopt` for this functionality."
        )
        raise e

    # warn user about usage of nonlinear constraints
    if domain.constraints:
        if np.any([isinstance(c, NonlinearConstraint) for c in domain.constraints]):
            warnings.warn(
                "Nonlinear constraints were detected. Not all features and checks are supported for this type of constraints. \
                Using them can lead to unexpected behaviour. Please make sure to provide jacobians for nonlinear constraints.",
                UserWarning,
            )

    # check that NChooseK constraints only impose an upper bound on the number of nonzero components (and no lower bound)
    assert all(
        c.min_count == 0
        for c in domain.constraints
        if isinstance(c, NChooseKConstraint)
    ), "NChooseKConstraint with min_count !=0 is not supported!"

    # determine number of experiments (only relevant if n_experiments is not provided by the user)
    n_experiments = get_n_experiments(
        domain=domain, model_type=model_type, n_experiments=n_experiments
    )

    #
    # Sampling initital values
    #

    if sampling is not None:
        sampling.sort_index(axis=1, inplace=True)
        x0 = sampling.values.flatten()
    else:
        if len(domain.constraints.get(NonlinearConstraint)) == 0:
            sampler = PolytopeSampler(
                data_model=PolytopeSamplerDataModel(domain=domain)
            )
            x0 = sampler.ask(n_experiments, return_all=False).to_numpy().flatten()
        else:
            warnings.warn(
                "Sampling failed. Falling back to uniform sampling on input domain.\
                          Providing a custom sampling strategy compatible with the problem can \
                          possibly improve performance."
            )
            x0 = (
                domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM)
                .to_numpy()
                .flatten()
            )

    # get objective function and its jacobian
    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )

    objective_class = get_objective_class(objective)
    d_optimality = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments, delta=delta
    )

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(
        domain, n_experiments, ignore_nchoosek=True
    )

    # find bounds imposing NChooseK constraints
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)

    # fix experiments if any are given
    if fixed_experiments is not None:
        fixed_experiments.sort_index(axis=1, inplace=True)
        domain.validate_candidates(fixed_experiments, only_inputs=True)
        for i, val in enumerate(fixed_experiments.values.flatten()):
            bounds[i] = (val, val)
            x0[i] = val

    # partially fix experiments if any are given
    bounds, x0 = partially_fix_experiment(
        bounds, fixed_experiments, n_experiments, partially_fixed_experiments, x0
    )

    # set ipopt options
    if ipopt_options is None:
        ipopt_options = {}
    _ipopt_options = {"maxiter": 500, "disp": 0}
    for key in ipopt_options.keys():
        _ipopt_options[key] = ipopt_options[key]
    if _ipopt_options["disp"] > 12:
        _ipopt_options["disp"] = 0

    #
    # Do the optimization
    #

    result = minimize_ipopt(
        d_optimality.evaluate,
        x0=x0,
        bounds=bounds,
        # "SLSQP" has no deeper meaning here and just ensures correct constraint standardization
        constraints=standardize_constraints(constraints, x0, "SLSQP"),
        options=_ipopt_options,
        jac=d_optimality.evaluate_jacobian,
    )

    design = pd.DataFrame(
        result["x"].reshape(n_experiments, len(domain.inputs)),
        columns=domain.inputs.get_keys(),
        index=[f"exp{i}" for i in range(n_experiments)],
    )

    # exit message
    if _ipopt_options[b"print_level"] > 12:  # type: ignore
        for key in ["fun", "message", "nfev", "nit", "njev", "status", "success"]:
            print(key + ":", result[key])
        X = model_formula.get_model_matrix(design).to_numpy()
        print("metrics:", metrics(X))

    # check if all points respect the domain and the constraint
    try:
        domain.validate_candidates(
            candidates=design.apply(lambda x: np.round(x, 8)),
            only_inputs=True,
            tol=1e-4,
        )
    except (ValueError, ConstraintNotFulfilledError):
        warnings.warn(
            "Some points do not lie inside the domain or violate constraints. Please check if the \
                results lie within your tolerance.",
            UserWarning,
        )

    return design


def partially_fix_experiment(
    bounds: list,
    fixed_experiments: Union[pd.DataFrame, None],
    n_experiments: int,
    partially_fixed_experiments: Union[pd.DataFrame, None],
    x0: np.ndarray,
) -> Tuple[List, np.ndarray]:
    """
    fixes some variables for experiments. Within one experiment not all variables need to be fixed.
    Variables can be fixed to one value or can be set to a range by setting a tuple with lower and upper bound
    Non-fixed variables have to be set to None or nan. Will also fix the experiments provided in fixed_experiments

    Args:
        bounds (list): current bounds
        fixed_experiments (pd.Dataframe): experiments which are guaranteed to be part of the design and are fully fixed
        n_experiments (int): number of experiments
        partially_fixed_experiments (pd.Dataframe): experiments which are partially fixed
        x0: initial guess

    Returns: Tuple of list and pd.Dataframe containing the new bounds for each variable and an adapted initial guess
        which comply with the bounds

    """

    shift = 0
    if partially_fixed_experiments is not None:
        partially_fixed_experiments.sort_index(axis=1, inplace=True)
        if fixed_experiments is not None:
            if (
                len(fixed_experiments) + len(partially_fixed_experiments)
                > n_experiments
            ):
                raise AttributeError(
                    "Number of fixed experiments and partially fixed experiments exceeds the number of total "
                    "experiments"
                )
            shift = len(fixed_experiments)

        shift = shift * len(partially_fixed_experiments.columns)
        for i, val in enumerate(np.array(partially_fixed_experiments.values).flatten()):
            index = shift + i
            if type(val) is tuple:
                bounds[index] = (val[0], val[1])
                x0[index] = val[0]
            elif val is not None and not np.isnan(val):
                bounds[index] = (val, val)
                x0[index] = val
    return bounds, x0


def check_fixed_experiments(
    domain: Domain, n_experiments: int, fixed_experiments: np.ndarray
) -> None:
    """Checks if the shape of the fixed experiments is correct and if the number of fixed experiments is valid
    Args:
        domain (Domain): domain defining the input variables used for the check.
        n_experiments (int): total number of experiments in the design that fixed_experiments are part of.
        fixed_experiments (np.ndarray): fixed experiment proposals to be checked.
    """

    n_fixed_experiments, D = np.array(fixed_experiments).shape

    if n_fixed_experiments >= n_experiments:
        raise ValueError(
            "For starting the optimization the total number of experiments must be larger that the number of fixed experiments."
        )

    if D != len(domain.inputs):
        raise ValueError(
            f"Invalid shape of fixed_experiments. Length along axis 1 is {D}, but must be {len(domain.inputs)}"
        )


def check_partially_and_fully_fixed_experiments(
    domain: Domain,
    n_experiments: int,
    fixed_experiments: np.ndarray,
    paritally_fixed_experiments: np.ndarray,
) -> None:
    """Checks if the shape of the fixed experiments is correct and if the number of fixed experiments is valid
    Args:
        domain (Domain): domain defining the input variables used for the check.
        n_experiments (int): total number of experiments in the design that fixed_experiments are part of.
        fixed_experiments (np.ndarray): fixed experiment proposals to be checked.
        paritally_fixed_experiments (np.ndarray): partially fixed experiment proposals to be checked.
    """

    check_fixed_experiments(domain, n_experiments, fixed_experiments)
    n_fixed_experiments, dim = np.array(fixed_experiments).shape

    n_partially_fixed_experiments, partially_dim = np.array(
        paritally_fixed_experiments
    ).shape

    if partially_dim != len(domain.inputs):
        raise ValueError(
            f"Invalid shape of partially_fixed_experiments. Length along axis 1 is {partially_dim}, but must be {len(domain.inputs)}"
        )

    if n_fixed_experiments + n_partially_fixed_experiments > n_experiments:
        warnings.warn(
            UserWarning(
                "The number of fixed experiments and partially fixed experiments exceeds the amount "
                "of the overall count of experiments. Partially fixed experiments may be cut of"
            )
        )


def get_n_experiments(
    domain: Domain, model_type: Union[str, Formula], n_experiments: Optional[int] = None
):
    """Determines a number of experiments which is appropriate for the model if no
    number is provided. Otherwise warns if the provided number of experiments is smaller than recommended.

    Args:
        domain (Domain): domain containing the model inputs.
        model_type (str, Formula): keyword or formulaic Formula describing the model.
        n_experiments (int, optional): number of experiments. Defaults to zero.

    Returns:
        n_experiments if an integer value for n_experiments is given. Number of model terms + 3 otherwise.

    """
    n_experiments_min = (
        len(
            get_formula_from_string(model_type=model_type, rhs_only=True, domain=domain)
        )
        + 3
    )

    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(
            f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}."
        )

    return n_experiments
