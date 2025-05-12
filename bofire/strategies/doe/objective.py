import warnings
from abc import abstractmethod
from copy import deepcopy
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import torch
from formulaic import Formula
from torch import Tensor
from torch.autograd.functional import hessian, jacobian

from bofire.data_models.constraints.api import (
    ConstraintNotFulfilledError,
    EqualityConstraint,
)
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.strategies.doe import (
    AOptimalityCriterion,
    DoEOptimalityCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    GOptimalityCriterion,
    IOptimalityCriterion,
    KOptimalityCriterion,
    OptimalityCriterion,
    SpaceFillingCriterion,
)
from bofire.data_models.types import Bounds
from bofire.strategies.doe.objective_base import Objective
from bofire.strategies.doe.utils import (
    _minimize,
    constraints_as_scipy_constraints,
    convert_formula_to_string,
    get_formula_from_string,
    nchoosek_constraints_as_bounds,
)
from bofire.utils.torch_tools import tkwargs


class ModelBasedObjective(Objective):
    def __init__(
        self,
        domain: Domain,
        formula: Formula,
        n_experiments: int,
        delta: float = 1e-7,
        transform_range: Optional[Bounds] = None,
    ) -> None:
        """Args:
        domain (Domain): A domain defining the DoE domain together with formula.
        formula (str or Formula): A formula containing all model terms.
        n_experiments (int): Number of experiments
        delta (float): A regularization parameter for the information matrix. Default value is 1e-7.
        transform_range (Bounds, optional): range to which the input variables are transformed before applying the objective function. Default is None.
        """
        super().__init__(
            domain=domain,
            n_experiments=n_experiments,
            delta=delta,
            transform_range=transform_range,
        )

        self.formula = deepcopy(formula)
        self.model_terms_string_expression = convert_formula_to_string(
            domain=domain, formula=formula
        )
        self.n_model_terms = len(list(np.array(formula, dtype=str)))

    def _evaluate(self, x: np.ndarray) -> float:
        D = torch.tensor(
            x.reshape(len(x.flatten()) // self.n_vars, self.n_vars),
            **tkwargs,
            requires_grad=False,
        )
        return float(self._evaluate_tensor(D))

    def _evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        D = torch.tensor(
            x.reshape(len(x.flatten()) // self.n_vars, self.n_vars),
            **tkwargs,
        )

        return (
            torch.tensor(jacobian(self._evaluate_tensor, D)).detach().numpy().flatten()
        )

    # FIXME: currently not returning the hessian in a way that is compatible with ipopt
    # also, the hessians of the constraints are missing
    def _evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        def _evaluate_from_flat_tensor(x: Tensor) -> Tensor:
            D = x.reshape(len(x.flatten()) // self.n_vars, self.n_vars)
            return self._evaluate_tensor(D)

        return (
            torch.tensor(
                hessian(_evaluate_from_flat_tensor, torch.tensor(x, **tkwargs))
            )
            .detach()
            .numpy()
        )

    def _evaluate_tensor(self, D: Tensor) -> Tensor:
        """Evaluate the objective function on the design matrix as a tensor."""
        var_dict = {var: D[:, i] for i, var in enumerate(self.vars)}
        var_dict["torch"] = torch  # type: ignore
        X = eval(str(self.model_terms_string_expression), {}, var_dict)
        return self._criterion(X)

    @abstractmethod
    def _criterion(self, X: Tensor) -> Tensor:
        """Function implementing the criterion acting on the design matrix X.

        Args:
            X (Tensor): Design matrix.

        Returns:
            A tensor containing a single number which is the value of the criterion.
        """
        pass


class IOptimality(ModelBasedObjective):
    """A class implementing the evaluation of I-criterion and its jacobian w.r.t.
    the inputs.
    """

    def __init__(
        self,
        domain: Domain,
        formula: Formula,
        n_experiments: int,
        delta: float = 1e-7,
        transform_range: Optional[Bounds] = None,
        n_space_filling_points: Optional[int] = None,
        ipopt_options: Optional[dict] = None,
    ) -> None:
        """
        Args:
            domain (Domain): A domain defining the DoE domain together with model_type.
            formula (str or Formula): A formula containing all model terms.
            n_experiments (int): Number of experiments
            delta (float): A regularization parameter for the information matrix. Default value is 1e-3.
            transform_range (Bounds, optional): range to which the input variables are transformed before applying the objective function. Default is None.
            n_space_filling_points (int, optional): Number of space filling points. Only relevant if SpaceFilling is used
            ipopt_options (dict, optional): Options for the Ipopt solver to generate space filling point.
                If None is provided, the default options (max_iter = 500) are used.
        """

        if transform_range is not None:
            raise ValueError(
                "IOptimality does not support transformations of the input variables."
            )

        super().__init__(
            domain=domain,
            formula=formula,
            n_experiments=n_experiments,
            delta=delta,
            transform_range=transform_range,
        )

        # uniformly fill the design space
        if np.any([isinstance(obj, EqualityConstraint) for obj in domain.constraints]):
            warnings.warn(
                "Equality constraints were detected. No equidistant grid of points can be generated. The design space will be filled via SpaceFilling.",
                UserWarning,
            )
            if n_space_filling_points is None:
                n_space_filling_points = n_experiments * 10

            x0 = (
                domain.inputs.sample(
                    n=n_space_filling_points, method=SamplingMethodEnum.UNIFORM
                )
                .to_numpy()
                .flatten()
            )
            objective_function = SpaceFilling(
                domain=domain,
                n_experiments=n_space_filling_points,
                delta=delta,
                transform_range=None,
            )
            constraints = constraints_as_scipy_constraints(
                domain, n_space_filling_points, ignore_nchoosek=True
            )
            bounds = nchoosek_constraints_as_bounds(domain, n_space_filling_points)

            x = _minimize(
                objective_function=objective_function,
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                ipopt_options={"print_level": 0},
                use_hessian=False,
            )

            self.Y = pd.DataFrame(
                x.reshape(n_space_filling_points, len(domain.inputs)),
                columns=domain.inputs.get_keys(),
                index=[f"gridpoint{i}" for i in range(n_space_filling_points)],
            )

        else:
            low, high = domain.inputs.get_bounds(specs={})
            points = [
                list(
                    np.linspace(
                        low[i],
                        high[i],
                        int(100 * (high[i] - low[i])),
                    )
                )
                for i in range(len(low))
            ]
            points = np.array(list(product(*points)))
            points = pd.DataFrame(points, columns=domain.inputs.get_keys())
            if len(domain.constraints) > 0:
                fulfilled = domain.constraints(experiments=points)
                fulfilled = np.array(
                    [
                        np.array(fulfilled.iloc[:, i]) <= 0.0
                        for i in range(fulfilled.shape[1])
                    ]
                )
                fulfilled = np.array(np.prod(fulfilled, axis=0), dtype=bool)
                self.Y = points[fulfilled]
            else:
                self.Y = points
            n_space_filling_points = len(self.Y)

        try:
            domain.validate_candidates(
                candidates=self.Y.apply(lambda x: np.round(x, 8)),
                only_inputs=True,
                tol=1e-4,
            )
        except (ValueError, ConstraintNotFulfilledError):
            warnings.warn(
                "Some points do not lie inside the domain or violate constraints. Please check if the \
                    results lie within your tolerance.",
                UserWarning,
            )

        X = formula.get_model_matrix(self.Y).to_numpy()
        self.YtY = torch.from_numpy(X.T @ X) / n_space_filling_points
        self.YtY.requires_grad = False

    def _criterion(self, X: Tensor) -> Tensor:
        return torch.trace(
            self.YtY.detach()
            @ torch.linalg.inv(X.T @ X + self.delta * torch.eye(self.n_model_terms))
        )


class DOptimality(ModelBasedObjective):
    """A class implementing the evaluation of logdet(X.T@X + delta), i.e. the
    D-optimality criterion
    """

    def _criterion(self, X: Tensor) -> Tensor:
        return -1 * torch.logdet(X.T @ X + self.delta * torch.eye(self.n_model_terms))


class AOptimality(ModelBasedObjective):
    """A class implementing the evaluation of tr((X.T@X + delta)^-1) and its jacobian w.r.t. the inputs.
    The jacobian evaluation is done analogously to DOptimality with the first part of the jacobian being
    the jacobian of tr((X.T@X + delta)^-1) instead of logdet(X.T@X + delta).
    """

    def _criterion(self, X: Tensor) -> Tensor:
        return torch.trace(
            torch.linalg.inv(X.T @ X + self.delta * torch.eye(self.n_model_terms))
        )


class GOptimality(ModelBasedObjective):
    """A class implementing the evaluation of max(diag(H)) and its jacobian w.r.t. the inputs where
    H = X @ (X.T@X + delta)^-1 @ X.T is the (regularized) hat matrix. The jacobian evaluation is done analogously
    to DOptimality with the first part of the jacobian being the jacobian of max(diag(H)) instead of
    logdet(X.T@X + delta).
    """

    def _criterion(self, X: Tensor) -> Tensor:
        return torch.max(
            torch.diag(
                X
                @ torch.linalg.inv(X.T @ X + self.delta * torch.eye(self.n_model_terms))
                @ X.T,
            ),
        )


class EOptimality(ModelBasedObjective):
    """A class implementing the evaluation of minus one times the minimum eigenvalue of (X.T @ X + delta)
    and its jacobian w.r.t. the inputs. The jacobian evaluation is done analogously to DOptimality with the
    first part of the jacobian being the jacobian of the smallest eigenvalue of (X.T @ X + delta) instead of
    logdet(X.T@X + delta).
    """

    def _criterion(self, X: Tensor) -> Tensor:
        return -1 * torch.min(
            torch.linalg.eigvalsh(X.T @ X + self.delta * torch.eye(self.n_model_terms)),
        )


class KOptimality(ModelBasedObjective):
    """A class implementing the evaluation of the condition number of (X.T @ X + delta)
    and its jacobian w.r.t. the inputs. The jacobian evaluation is done analogously to
    DOptimality with the first part of the jacobian being the jacobian of condition number
    of (X.T @ X + delta) instead of logdet(X.T@X + delta).
    """

    def _criterion(self, X: Tensor) -> Tensor:
        return torch.linalg.cond(
            X.T @ X + self.delta * torch.eye(self.n_model_terms),
        )


class SpaceFilling(Objective):
    def _evaluate(self, x: np.ndarray) -> float:
        X = self._convert_input_to_tensor(x, requires_grad=False)
        return float(
            -torch.sum(torch.sort(torch.pdist(X.detach()))[0][: self.n_experiments]),
        )

    def _evaluate_jacobian(self, x: np.ndarray) -> float:  # type: ignore
        X = self._convert_input_to_tensor(x, requires_grad=True)
        torch.sum(torch.sort(torch.pdist(X))[0][: self.n_experiments]).backward()

        return -X.grad.detach().numpy().flatten()  # type: ignore

    def _convert_input_to_tensor(
        self,
        x: np.ndarray,
        requires_grad: bool = True,
    ) -> Tensor:
        X = pd.DataFrame(
            x.reshape(len(x.flatten()) // self.n_vars, self.n_vars),
            columns=self.vars,
        )
        return torch.tensor(X.values, requires_grad=requires_grad, **tkwargs)


def get_objective_function(
    criterion: Optional[OptimalityCriterion],
    domain: Domain,
    n_experiments: int,
    inputs_for_formula: Optional[Inputs] = None,
) -> Objective:
    if isinstance(criterion, DoEOptimalityCriterion):
        _inputs_for_formula = (
            domain.inputs if inputs_for_formula is None else inputs_for_formula
        )
        if isinstance(criterion, DOptimalityCriterion):
            return DOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
            )
        if isinstance(criterion, AOptimalityCriterion):
            return AOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
            )
        if isinstance(criterion, GOptimalityCriterion):
            return GOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
            )
        if isinstance(criterion, EOptimalityCriterion):
            return EOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
            )
        if isinstance(criterion, KOptimalityCriterion):
            return KOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
            )
        if isinstance(criterion, IOptimalityCriterion):
            return IOptimality(
                domain,
                formula=get_formula_from_string(
                    criterion.formula,
                    inputs=_inputs_for_formula,
                ),
                n_experiments=n_experiments,
                delta=criterion.delta,
                transform_range=criterion.transform_range,
                n_space_filling_points=criterion.n_space_filling_points,
                ipopt_options=criterion.ipopt_options,
            )
    if isinstance(criterion, SpaceFillingCriterion):
        return SpaceFilling(
            domain,
            n_experiments=n_experiments,
            delta=criterion.delta,
            transform_range=criterion.transform_range,
        )
    else:
        raise NotImplementedError("Criterion type not implemented!")
