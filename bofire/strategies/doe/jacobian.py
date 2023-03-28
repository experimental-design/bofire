from copy import deepcopy
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import scipy as sp
from formulaic import Formula

from bofire.data_models.domain.api import Domain


class JacobianForLogdet:
    """A class representing the jacobian/gradient of logdet(X.T@X) w.r.t. the inputs.
    It can be divided into two parts, one for logdet(X.T@X) w.r.t. X (there is a simple
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
        """
        Args:
            domain (Domain): An opti domain defining the DoE domain together with model_type.
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

        self.model_jacobian_t = get_model_jacobian_t(self.vars, model)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Computes the full jacobian for the given input."""

        # get model matrix X
        x = np.array(x)
        x = x.reshape(self.n_experiments, self.n_vars)
        X = pd.DataFrame(x, columns=self.vars)
        X = self.model.get_model_matrix(X).to_numpy()

        # first part of jacobian
        J1 = (
            -2
            * sp.linalg.solve(
                X.T @ X + self.delta * np.eye(self.n_model_terms), X.T, assume_a="pos"
            ).T
        )
        J1 = np.repeat(J1, self.n_vars, axis=0).reshape(
            self.n_experiments, self.n_vars, self.n_model_terms
        )

        # second part of jacobian
        J2 = self.model_jacobian_t(x)

        # combine both parts
        J = J1 * J2
        J = np.sum(J, axis=-1)

        return J.flatten()


def default_jacobian_building_block(
    vars: List[str], model_terms: List[str]
) -> Callable:
    """Returns a function that returns the terms of the reduced jacobian for one experiment.

    Args:
        vars (List[str]): List of variable names of the model
        model_terms (List[str]): List of model terms saved as string.

    Returns:
        A function that returns a jacobian building block usable for models up to second order.
    """

    n_vars = len(vars)

    # find the term names
    terms = ["1"]
    for name in vars:
        terms.append(name)

    for name in vars:
        terms.append(name + "**2")
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            term = str(Formula(vars[j] + ":" + vars[i] + "-1"))
            terms.append(term)

    for name in vars:
        terms.append(name + "**3")
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            for k in range(j + 1, n_vars):
                term = str(Formula(vars[k] + ":" + vars[j] + ":" + vars[i] + "-1"))
                terms.append(term)

    def jacobian_building_block(x: np.ndarray) -> np.ndarray:
        """Computes the jacobian building block for a single experiment with inputs x."""
        x = np.array(x)
        B = np.zeros(shape=(n_vars, len(terms)))

        # derivatives of intercept term are zero

        # derivatives of linear terms
        B[:, 1 : n_vars + 1] = np.eye(n_vars)

        # derivatives of quadratic terms
        B[:, n_vars + 1 : 2 * n_vars + 1] = 2 * np.diag(x)

        # derivatives of second order interaction terms
        col = 2 * n_vars + 1
        for i, _ in enumerate(vars[:-1]):
            n_terms = len(vars[i + 1 :])
            B[i, col : col + n_terms] = x[i + 1 :]
            B[i + 1 :, col : col + n_terms] = x[i] * np.eye(n_terms)
            col += n_terms

        # derivatives of third order terms
        B[:, col : col + n_vars] = 3 * np.diag(x**2)
        col += n_vars

        # derivatives of third order interaction terms
        for i, _ in enumerate(vars[:-2]):
            for j in range(i + 1, n_vars - 1):
                n_terms = len(vars[j + 1 :])
                B[i, col : col + n_terms] = x[j] * x[j + 1 :]
                B[j, col : col + n_terms] = x[i] * x[j + 1 :]
                B[j + 1 :, col : col + n_terms] = x[i] * x[j] * np.eye(n_terms)
                col += n_terms

        return pd.DataFrame(B, columns=terms)[model_terms].to_numpy()

    return jacobian_building_block


def get_model_jacobian_t(vars: List[str], formula: Formula) -> Callable:
    """Returns a function that computes the transpose of the jacobian of the model
    where each term of the model is viewed as one component of a vector valued function.

    Args:
        vars (List[str]): List of variable names of the model
        model_terms (List[str]): List of model terms saved as string.

    Returns:
        A function that returns a jacobian building block for a given model
    """

    """Computes the jacobian building blocks for all experiments in x."""

    terms_jacobian_t = []
    for var in vars:
        terms_jacobian_t.append(
            [
                str(term).replace(":", "*") + f" + 0 * {vars[0]}"
                for term in formula.differentiate(var, use_sympy=True)
            ]
        )  # 0*vars[0] added to make sure terms are evaluated as series, not as number

    def model_jacobian_t(x: np.ndarray) -> np.ndarray:
        """Computes the transpose of the model jacobian for each experiment in input x."""
        X = pd.DataFrame(x, columns=vars)
        X.eval(terms_jacobian_t)
        jacobians = np.swapaxes(X.eval(terms_jacobian_t), 0, 2)
        return np.swapaxes(jacobians, 1, 2)

    return model_jacobian_t
