import warnings
from itertools import combinations_with_replacement, product
from queue import PriorityQueue
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.optimize._minimize import standardize_constraints

from bofire.data_models.constraints.api import NChooseKConstraint, NonlinearConstraint
from bofire.data_models.constraints.constraint import ConstraintNotFulfilledError
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    ContinuousBinaryInput,
    ContinuousDiscreteInput,
    Input,
)
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
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
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
            objective (OptimalityCriterionEnum): OptimalityCriterionEnum object indicating which objective function to use.
            verbose (bool): if true, print information during the optimization process
        Returns:
            A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
            local optimum.
    """
    from bofire.strategies.doe.branch_and_bound import NodeExperiment, bnb

    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(objective)
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments, delta=delta
    )

    binary_vars = domain.get_features(ContinuousBinaryInput)
    domain.get_features(includes=[Input], excludes=ContinuousBinaryInput)

    for var in binary_vars:
        var.relax()

    column_keys = domain.inputs.get_keys()

    initial_branch = pd.DataFrame(
        np.full((n_experiments, len(column_keys)), None), columns=column_keys
    )
    initial_design = find_local_max_ipopt(
        domain,
        model_type,
        n_experiments,
        delta,
        ipopt_options,
        sampling,
        fixed_experiments,
        partially_fixed_experiments=initial_branch,
        objective=objective,
    )
    initial_value = objective_class.evaluate(
        initial_design.to_numpy().flatten(),
    )

    discrete_vars = domain.inputs.get(includes=ContinuousDiscreteInput)
    initial_node = NodeExperiment(
        initial_branch,
        initial_design,
        initial_value,
        domain.categorical_groups,
        discrete_vars,
    )

    initial_queue = PriorityQueue()
    initial_queue.put(initial_node)

    result_node = bnb(
        initial_queue,
        domain=domain,
        model_type=model_type,
        n_experiments=n_experiments,
        delta=delta,
        ipopt_options=ipopt_options,
        sampling=sampling,
        fixed_experiments=fixed_experiments,
        objective=objective,
    )

    return result_node.design_matrix


def find_local_max_ipopt_binary_naive(
    domain: Domain,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    delta: float = 1e-7,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
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
            verbose (bool): if true, print information during the optimization process
        Returns:
            A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
            local optimum.
    """
    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )
    objective_class = get_objective_class(objective)
    objective_class = objective_class(
        domain=domain, model=model_formula, n_experiments=n_experiments, delta=delta
    )

    binary_vars = domain.get_features(ContinuousBinaryInput)
    domain.get_features(includes=[Input], excludes=ContinuousBinaryInput)
    list_keys = binary_vars.get_keys()

    for var in binary_vars:
        var.relax()

    allowed_fixations = []
    for group in domain.categorical_groups:
        allowed_fixations.append(np.eye(len(group)))

    allowed_fixations = product(*allowed_fixations)
    all_n_fixed_experiments = combinations_with_replacement(
        allowed_fixations, n_experiments
    )

    column_keys = domain.inputs.get_keys()

    minimum = float("inf")
    optimal_design = None
    number_of_non_binary_vars = len(domain.inputs) - len(binary_vars)
    for i, binary_fixed_experiments in enumerate(list(all_n_fixed_experiments)):
        binary_fixed_experiments = np.array(
            [
                var
                for experiment in binary_fixed_experiments
                for group in experiment
                for var in group
            ]
        ).reshape(n_experiments, len(binary_vars))

        one_set_of_experiments = np.concatenate(
            [
                binary_fixed_experiments,
                np.full((n_experiments, number_of_non_binary_vars), None),
            ],
            axis=1,
        )

        one_set_of_experiments = pd.DataFrame(
            one_set_of_experiments, columns=column_keys
        )

        if sampling is not None:
            sampling.loc[:, list_keys] = one_set_of_experiments[list_keys].to_numpy()
        try:
            current_design = find_local_max_ipopt(
                domain,
                model_type,
                n_experiments,
                delta,
                ipopt_options,
                sampling,
                fixed_experiments,
                one_set_of_experiments,
                objective,
            )

            temp_value = objective_class.evaluate(
                current_design.to_numpy().flatten(),
            )
            if minimum is None or minimum > temp_value:
                minimum = temp_value
                optimal_design = current_design
            if verbose:
                print(
                    f"branch: {i}, solution: {temp_value}, minimum after run {minimum}, difference: {temp_value - minimum}"
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
            If the parameter fixed_experiments is given, those experiments will be prioritized.
            Non-fixed variables have to be set to None.
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
        domain.validate_candidates(sampling, only_inputs=True)
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
        domain.validate_candidates(fixed_experiments, only_inputs=True)
        fixed_experiments = np.array(fixed_experiments.values)
        for i, val in enumerate(fixed_experiments.flatten()):
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
    except ValueError:
        warnings.warn(
            "Some points do not lie inside the domain or violate constraints. Please check if the \
                results lie within your tolerance.",
            UserWarning,
        )

    return design


def partially_fix_experiment(
    bounds, fixed_experiments, n_experiments, partially_fixed_experiments, x0
):
    if partially_fixed_experiments is not None:
        if fixed_experiments is not None:
            cut_of_k_experiments = (
                len(fixed_experiments)
                + len(partially_fixed_experiments)
                - n_experiments
            )
        else:
            cut_of_k_experiments = len(partially_fixed_experiments) - n_experiments
        if cut_of_k_experiments > 0:
            cut_of_start_index = len(partially_fixed_experiments) - cut_of_k_experiments
            cut_of_end_index = len(partially_fixed_experiments)
            partially_fixed_experiments.drop(
                list(range(cut_of_start_index, cut_of_end_index)), inplace=True
            )

        for i, val in enumerate(np.array(partially_fixed_experiments.values).flatten()):
            if type(val) is tuple:
                bounds[i] = (val[0], val[1])
                x0[i] = val[0]
            elif val is not None:
                bounds[i] = (val, val)
                x0[i] = val
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
