import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.optimize._minimize import standardize_constraints

from bofire.data_models.constraints.api import NChooseKConstraint, NonlinearConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
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


def find_local_max_ipopt(
    domain: Domain,
    model_type: Union[str, Formula],
    n_experiments: Optional[int] = None,
    delta: float = 1e-7,
    ipopt_options: Dict = {},
    sampling: Optional[pd.DataFrame] = None,
    fixed_experiments: Optional[pd.DataFrame] = None,
    objective: OptimalityCriterionEnum = OptimalityCriterionEnum.D_OPTIMALITY,
) -> pd.DataFrame:
    """Function computing a d-optimal design" for a given domain and model.
    Args:
        domain (Domain): domain containing the inputs and constraints.
        model_type (str, Formula): keyword or formulaic Formula describing the model. Known keywords
            are "linear", "linear-and-interactions", "linear-and-quadratic", "fully-quadratic".
        n_experiments (int): Number of experiments. By default the value corresponds to
            the number of model terms - dimension of ker() + 3.
        delta (float): Regularization parameter. Default value is 1e-3.
        ipopt_options (Dict): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html)
        sampling (Sampling, np.ndarray): Sampling class or a np.ndarray object containing the initial guess.
        fixed_experiments (pd.DataFrame): dataframe containing experiments that will be definitely part of the design.
            Values are set before the optimization.
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
        [
            c.min_count == 0
            for c in domain.constraints
            if isinstance(c, NChooseKConstraint)
        ]
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
        x0 = sampling.values
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

    # set ipopt options
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
