import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from formulaic import Formula

from bofire.data_models.constraints.api import (
    ConstraintNotFulfilledError,
    NChooseKConstraint,
    NonlinearConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.doe.objective import Objective
from bofire.strategies.doe.utils import (
    _minimize,
    constraints_as_scipy_constraints,
    nchoosek_constraints_as_bounds,
)
from bofire.strategies.random import RandomStrategy


def find_local_max_ipopt(
    domain: Domain,
    objective_function: Objective,
    ipopt_options: Optional[Dict] = None,
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    partially_fixed_experiments: Optional[pd.DataFrame] = None,
    use_hessian: bool = False,
    use_cyipopt: Optional[bool] = None,
) -> pd.DataFrame:
    """Function computing an optimal design for a given domain and model.

    Args:
        domain: domain containing the inputs and constraints.
        objective_function: The function defining the objective of the optimizattion.
        ipopt_options: options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
        sampling : dataframe containing the initial guess.
        fixed_experiments : dataframe containing experiments that will be definitely part of the design.
            Values are set before the optimization.
        partially_fixed_experiments: dataframe containing (some) fixed variables for experiments.
            Values are set before the optimization. Within one experiment not all variables need to be fixed.
            Variables can be fixed to one value or can be set to a range by setting a tuple with lower and upper bound
            Non-fixed variables have to be set to None or nan.
        use_hessian: If True, the hessian of the objective function is used. Default is False.
        use_cyipopt: If True, cyipopt is used, otherwise scipy.minimize(). Default is None.
            If None, cyipopt is used if available.

    Returns:
        A pd.DataFrame object containing the best found input for the experiments. In general, this is only a
        local optimum.

    """
    #
    # Checks and preparation steps
    #
    n_experiments = objective_function.n_experiments
    if partially_fixed_experiments is not None:
        # check if partially fixed experiments are valid
        check_partially_fixed_experiments(
            domain,
            n_experiments,
            partially_fixed_experiments,
        )
        # no columns from partially fixed experiments which are not in the domain
        partially_fixed_experiments = partially_fixed_experiments[
            domain.inputs.get_keys()
        ]

    if fixed_experiments is not None:
        # check if  fixed experiments are valid
        check_fixed_experiments(domain, n_experiments, fixed_experiments)
        # no columns from fixed experiments which are not in the domain
        fixed_experiments = fixed_experiments[domain.inputs.get_keys()]

    if (partially_fixed_experiments is not None) and (fixed_experiments is not None):
        # check if partially fixed experiments and fixed experiments are valid
        check_partially_and_fully_fixed_experiments(
            domain,
            n_experiments,
            fixed_experiments,
            partially_fixed_experiments,
        )

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

    #
    # Sampling initial values
    #

    if sampling is not None:
        sampling.sort_index(axis=1, inplace=True)
        x0 = sampling.values.flatten()
    try:
        sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
        x0 = (
            sampler.ask(n_experiments, raise_validation_error=False)
            .to_numpy()
            .flatten()
        )
    except Exception as e:
        warnings.warn(str(e))
        warnings.warn(
            "Sampling failed. Falling back to uniform sampling on input domain.\
                      Providing a custom sampling strategy compatible with the problem can \
                      possibly improve performance.",
        )
        x0 = (
            domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM)
            .to_numpy()
            .flatten()
        )

    # write constraints as scipy constraints
    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=True,
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
        bounds,
        fixed_experiments,
        n_experiments,
        partially_fixed_experiments,
        x0,
    )

    # set ipopt options
    if ipopt_options is None:
        ipopt_options = {}
    _ipopt_options = {"max_iter": 500, "print_level": 0}
    for key in ipopt_options.keys():
        _ipopt_options[key] = ipopt_options[key]
    if _ipopt_options["print_level"] > 12:
        _ipopt_options["print_level"] = 0

    #
    # Do the optimization
    #
    x = _minimize(
        objective_function=objective_function,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        use_hessian=use_hessian,
        ipopt_options=_ipopt_options,
        use_cyipopt=use_cyipopt,
    )

    design = pd.DataFrame(
        x.reshape(n_experiments, len(domain.inputs)),
        columns=domain.inputs.get_keys(),
        index=[f"exp{i}" for i in range(n_experiments)],
    )
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
    """Fixes some variables for experiments. Within one experiment not all variables need to be fixed.
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
                    "experiments",
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
    domain: Domain,
    n_experiments: int,
    fixed_experiments: pd.DataFrame,
) -> None:
    """Checks if the shape of the fixed experiments is correct and if the number of fixed experiments is valid
    Args:
        domain (Domain): domain defining the input variables used for the check.
        n_experiments (int): total number of experiments in the design that fixed_experiments are part of.
        fixed_experiments (pd.DataFrame): fixed experiment proposals to be checked.
    """
    n_fixed_experiments = len(fixed_experiments.index)

    if n_fixed_experiments >= n_experiments:
        raise ValueError(
            "For starting the optimization the total number of experiments must be larger that the number of fixed experiments.",
        )

    domain.validate_candidates(
        candidates=fixed_experiments,
        only_inputs=True,
    )


def check_partially_fixed_experiments(
    domain: Domain,
    n_experiments: int,
    partially_fixed_experiments: pd.DataFrame,
) -> None:
    n_partially_fixed_experiments = len(partially_fixed_experiments.index)

    # for partially fixed experiments only check if all inputs are part of the domain
    if not all(
        key in partially_fixed_experiments.columns for key in domain.inputs.get_keys()
    ):
        raise ValueError(
            "Domain contains inputs that are not part of partially fixed experiments. Every input must be present as a column.",
        )

    if n_partially_fixed_experiments > n_experiments:
        warnings.warn(
            UserWarning(
                "The number of partially fixed experiments exceeds the amount "
                "of the overall count of experiments. Partially fixed experiments may be cut off",
            ),
        )


def check_partially_and_fully_fixed_experiments(
    domain: Domain,
    n_experiments: int,
    fixed_experiments: pd.DataFrame,
    partially_fixed_experiments: pd.DataFrame,
) -> None:
    """Checks if the shape of the fixed experiments is correct and if the number of fixed experiments is valid
    Args:
        domain (Domain): domain defining the input variables used for the check.
        n_experiments (int): total number of experiments in the design that fixed_experiments are part of.
        fixed_experiments (pd.DataFrame): fixed experiment proposals to be checked.
        partially_fixed_experiments (pd.DataFrame): partially fixed experiment proposals to be checked.
    """
    check_fixed_experiments(domain, n_experiments, fixed_experiments)
    check_partially_fixed_experiments(
        domain,
        n_experiments,
        partially_fixed_experiments,
    )
    n_fixed_experiments = len(fixed_experiments.index)

    n_partially_fixed_experiments = len(partially_fixed_experiments.index)

    if n_fixed_experiments + n_partially_fixed_experiments > n_experiments:
        warnings.warn(
            UserWarning(
                "The number of fixed experiments and partially fixed experiments exceeds the amount "
                "of the overall count of experiments. Partially fixed experiments may be cut off",
            ),
        )


def get_n_experiments(model_type: Formula, n_experiments: Optional[int] = None):
    """Determines a number of experiments which is appropriate for the model if no
    number is provided. Otherwise warns if the provided number of experiments is smaller than recommended.

    Args:
        domain (Domain): domain containing the model inputs.
        model_type (str, Formula): keyword or formulaic Formula describing the model.
        n_experiments (int, optional): number of experiments. Defaults to zero.

    Returns:
        n_experiments if an integer value for n_experiments is given. Number of model terms + 3 otherwise.

    """
    n_experiments_min = len(model_type) + 3

    if n_experiments is None:
        n_experiments = n_experiments_min
    elif n_experiments < n_experiments_min:
        warnings.warn(
            f"The minimum number of experiments is {n_experiments_min}, but the current setting is n_experiments={n_experiments}.",
        )

    return n_experiments
