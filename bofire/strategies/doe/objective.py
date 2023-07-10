from abc import abstractmethod
from copy import deepcopy
from typing import Type

import numpy as np
import pandas as pd
import torch
from formulaic import Formula
from torch import Tensor

from bofire.data_models.domain.api import Domain
from bofire.strategies.enum import OptimalityCriterionEnum
from bofire.utils.torch_tools import tkwargs


class Objective:
    def __init__(
        self,
        domain: Domain,
        model: Formula,
        n_experiments: int,
        delta: float = 1e-6,
    ) -> None:
        """
        Args:
            domain (Domain): A domain defining the DoE domain together with model_type.
            model_type (str or Formula): A formula containing all model terms.
            n_experiments (int): Number of experiments
            delta (float): A regularization parameter for the information matrix. Default value is 1e-3.

        """

        self.model = deepcopy(model)
        self.domain = deepcopy(domain)
        self.n_experiments = n_experiments
        self.delta = delta

        self.vars = self.domain.inputs.get_keys()
        self.n_vars = len(self.domain.inputs)

        self.model_terms = list(np.array(model, dtype=str))
        self.n_model_terms = len(self.model_terms)

        # terms for model jacobian
        self.terms_jacobian_t = []
        for var in self.vars:
            _terms = [
                str(term).replace(":", "*") + f" + 0 * {self.vars[0]}"
                for term in model.differentiate(var, use_sympy=True)
            ]  # 0*vars[0] added to make sure terms are evaluated as series, not as number
            terms = "["
            for t in _terms:
                terms += t + ", "
            terms = terms[:-1] + "]"

            self.terms_jacobian_t.append(terms)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        pass

    def _convert_input_to_model_tensor(
        self, x: np.ndarray, requires_grad: bool = True
    ) -> Tensor:
        """

        Args:
            x: x (np.ndarray): values of design variables a 1d array.
        """
        assert x.ndim == 1, "values of design should be 1d array"
        X = pd.DataFrame(
            x.reshape(len(x.flatten()) // self.n_vars, self.n_vars), columns=self.vars
        )
        X = self.model.get_model_matrix(X)
        return torch.tensor(X.values, requires_grad=requires_grad, **tkwargs)

    def _model_jacobian_t(self, x: np.ndarray) -> np.ndarray:
        """Computes the transpose of the model jacobian for each experiment in input x."""
        X = pd.DataFrame(x.reshape(self.n_experiments, self.n_vars), columns=self.vars)
        jacobians = np.swapaxes(X.eval(self.terms_jacobian_t), 0, 2)
        return np.swapaxes(jacobians, 1, 2)


class DOptimality(Objective):
    """A class implementing the evaluation of logdet(X.T@X + delta) and its jacobian w.r.t. the inputs.
    The Jacobian can be divided into two parts, one for logdet(X.T@ + delta) w.r.t. X (there is a simple
    closed expression for this one) and one model dependent part for the jacobian of X.T@X
    w.r.t. the inputs. Because each row of X only depends on the inputs of one experiment
    the second part can be formulated in a simplified way. It is built up with n_experiment
    blocks of the same structure which is represended by the attribute jacobian_building_block.

    A nice derivation for the "first part" of the jacobian can be found [here](https://angms.science/doc/LA/logdet.pdf).
    The second part consists of the partial derivatives of the model terms with
    respect to the inputs. We denote the value of the i-th model term from the j-th experiment
    with y_ij and the i-th input value of the j-th experiment with x_ij. N stands for the number
    of model terms, n for the number of input terms and M for the number of experiments.
    Here, we only consider models up to second order, but the computation can easily be extended
    for higher-ordermodels.

    To do the computation in the most basic way, we could compute the partial derivative of every
    single model term and experiment with respect to every single input and experiment. We could write
    this in one large matrix and multiply the first part of the gradient as a long vector from the right
    side.
    But because of the structure of the domain we can do the same computation with much smaller
    matrices:
    First, we write the first part of the jacobian as the matrix (df/dy_ij)_ij where i goes from 1 to N
    and j goes from 1 to M.

    Second, we compute a rank 3 tensor (K_kij)_kij. k goes from 1 to M, i from 1 to n and j from 1 to N.
    For each k (K_kij)_ij contains the partial derivatives (dy_jk/dx_ik)_ij. Note that the values of the
    entries of (dy_jk/dx_ik)_ij only depend on the input values of the k-th experiment. The function
    default_jacobian_building_block implements the computation of these matrices/"building blocks".

    Then, we notice that the model term values of the j-th experiment only depend on the input values of
    the j-th experiment. Thus, to compute the partial derivative df/dx_ik we only have to compute the euclidian
    scalar product of (K_kij)_j and (df/dy_jk)_j. The way how we built the two parts of the jacobian allows us
    to compute this scalar product in a vectorized way for all x_ik at once, see also JacobianForLogDet.jacobian.
    """

    def __init__(
        self,
        domain: Domain,
        model: Formula,
        n_experiments: int,
        delta: float = 1e-7,
    ) -> None:
        super().__init__(
            domain=domain, model=model, n_experiments=n_experiments, delta=delta
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Computes the minus one times the sum of the log of the eigenvalues of X.T @ X + delta.
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            -log(det(X.T@X+delta))

        """
        X = self._convert_input_to_model_tensor(x, requires_grad=False)
        return float(
            -1
            * torch.logdet(
                X.detach().T @ X.detach() + self.delta * torch.eye(self.n_model_terms)
            )
        )

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the jacobian of minus one times the log of the determinant of X.T @ X + delta.
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            The jacobian of -log(det(X.T@X+delta)) as numpy array
        """
        # get model matrix X
        X = self._convert_input_to_model_tensor(x, requires_grad=True)

        # first part of jacobian
        torch.logdet(X.T @ X + self.delta * torch.eye(self.n_model_terms)).backward()
        J1 = -1 * X.grad.detach().numpy()  # type: ignore
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self._model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


class AOptimality(Objective):
    """A class implementing the evaluation of tr((X.T@X + delta)^-1) and its jacobian w.r.t. the inputs.
    The jacobian evaluation is done analogously to DOptimality with the first part of the jacobian being
    the jacobian of tr((X.T@X + delta)^-1) instead of logdet(X.T@X + delta).
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Computes the trace of the inverse of X.T @ X + delta.
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            tr((X.T@X+delta)^-1)

        """
        X = self._convert_input_to_model_tensor(x, requires_grad=False)
        return float(
            torch.trace(
                torch.linalg.inv(
                    X.detach().T @ X.detach()
                    + self.delta * torch.eye(self.n_model_terms)
                )
            )
        )

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the jacobian of the trace of the inverse of X.T @ X + delta.
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            The jacobian of tr((X.T@X+delta)^-1) as numpy array
        """
        # get model matrix X
        X = self._convert_input_to_model_tensor(x, requires_grad=True)

        # first part of jacobian
        torch.trace(
            torch.linalg.inv(X.T @ X + self.delta * torch.eye(self.n_model_terms))
        ).backward()
        J1 = X.grad.detach().numpy()  # type: ignore
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self._model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


class GOptimality(Objective):
    """A class implementing the evaluation of max(diag(H)) and its jacobian w.r.t. the inputs where
    H = X @ (X.T@X + delta)^-1 @ X.T is the (regularized) hat matrix. The jacobian evaluation is done analogously
    to DOptimality with the first part of the jacobian being the jacobian of max(diag(H)) instead of
    logdet(X.T@X + delta).
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Computes the maximum diagonal entry of H = X @ (X.T@X + delta)^-1 @ X.T .
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            max(diag(H))

        """
        X = self._convert_input_to_model_tensor(x, requires_grad=False)
        H = (
            X.detach()
            @ torch.linalg.inv(
                X.detach().T @ X.detach() + self.delta * torch.eye(self.n_model_terms)
            )
            @ X.detach().T
        )
        return float(torch.max(torch.diag(H)))

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the jacobian of the maximum diagonal element of H = X @ (X.T @ X + delta)^-1 @ X.T.
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            The jacobian of max(diag(H)) as numpy array
        """
        # get model matrix X
        X = self._convert_input_to_model_tensor(x, requires_grad=True)

        # first part of jacobian
        torch.max(
            torch.diag(
                X
                @ torch.linalg.inv(X.T @ X + self.delta * torch.eye(self.n_model_terms))
                @ X.T
            )
        ).backward()
        J1 = X.grad.detach().numpy()  # type: ignore
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self._model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


class EOptimality(Objective):
    """A class implementing the evaluation of minus one times the minimum eigenvalue of (X.T @ X + delta)
    and its jacobian w.r.t. the inputs. The jacobian evaluation is done analogously to DOptimality with the
    first part of the jacobian being the jacobian of the smallest eigenvalue of (X.T @ X + delta) instead of
    logdet(X.T@X + delta).
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Computes minus one times the minimum eigenvalue of (X.T@X + delta).
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            min(eigvals(X.T @ X + delta))
        """
        X = self._convert_input_to_model_tensor(x, requires_grad=False)
        return -1 * float(
            torch.min(
                torch.linalg.eigvalsh(
                    X.detach().T @ X.detach()
                    + self.delta * torch.eye(self.n_model_terms)
                )
            )
        )

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the jacobian of minus one times the minimum eigenvalue of (X.T @ X + delta).
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            The jacobian of -1 * min(eigvals(X.T @ X + delta)) as numpy array
        """
        # get model matrix X
        X = self._convert_input_to_model_tensor(x, requires_grad=True)

        # first part of jacobian
        torch.min(
            torch.linalg.eigvalsh(X.T @ X + self.delta * torch.eye(self.n_model_terms))
        ).backward()
        J1 = -1 * X.grad.detach().numpy()  # type: ignore
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self._model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


class KOptimality(Objective):
    """A class implementing the evaluation of the condition number of (X.T @ X + delta)
    and its jacobian w.r.t. the inputs. The jacobian evaluation is done analogously to
    DOptimality with the first part of the jacobian being the jacobian of condition number
    of (X.T @ X + delta) instead of logdet(X.T@X + delta).
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Computes condition number of (X.T@X + delta).
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            cond(X.T @ X + delta)
        """
        X = self._convert_input_to_model_tensor(x, requires_grad=False)
        return float(
            torch.linalg.cond(
                X.detach().T @ X.detach() + self.delta * torch.eye(self.n_model_terms)
            )
        )

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the jacobian of the condition number of (X.T @ X + delta).
        Where X is the model matrix corresponding to x.

        Args:
            x (np.ndarray): values of design variables a 1d array.

        Returns:
            The jacobian of cond(X.T @ X + delta) as numpy array
        """
        # get model matrix X
        X = self._convert_input_to_model_tensor(x, requires_grad=True)

        # first part of jacobian
        torch.linalg.cond(
            X.T @ X + self.delta * torch.eye(self.n_model_terms)
        ).backward()
        J1 = X.grad.detach().numpy()  # type: ignore
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self._model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


class SpaceFilling(Objective):
    def evaluate(self, x: np.ndarray) -> float:
        X = self._convert_input_to_tensor(x, requires_grad=False)
        return float(
            -torch.sum(torch.sort(torch.pdist(X.detach()))[0][: self.n_experiments])
        )

    def evaluate_jacobian(self, x: np.ndarray) -> float:
        X = self._convert_input_to_tensor(x, requires_grad=True)
        torch.sum(torch.sort(torch.pdist(X))[0][: self.n_experiments]).backward()

        return -X.grad.detach().numpy().flatten()  # type: ignore

    def _convert_input_to_tensor(
        self, x: np.ndarray, requires_grad: bool = True
    ) -> Tensor:
        X = pd.DataFrame(
            x.reshape(len(x.flatten()) // self.n_vars, self.n_vars), columns=self.vars
        )
        return torch.tensor(X.values, requires_grad=requires_grad, **tkwargs)


def get_objective_class(objective: OptimalityCriterionEnum) -> Type:
    objective = OptimalityCriterionEnum(objective)

    if objective == OptimalityCriterionEnum.D_OPTIMALITY:
        return DOptimality
    elif objective == OptimalityCriterionEnum.A_OPTIMALITY:
        return AOptimality
    elif objective == OptimalityCriterionEnum.G_OPTIMALITY:
        return GOptimality
    elif objective == OptimalityCriterionEnum.E_OPTIMALITY:
        return EOptimality
    elif objective == OptimalityCriterionEnum.K_OPTIMALITY:
        return KOptimality
    elif objective == OptimalityCriterionEnum.SPACE_FILLING:
        return SpaceFilling
