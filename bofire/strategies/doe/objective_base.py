from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np

from bofire.data_models.domain.api import Domain
from bofire.data_models.types import Bounds
from bofire.strategies.doe.transform import IndentityTransform, MinMaxTransform


class Objective:
    """Base class for objectives in the context of Design of Experiments (DoE)."""

    def __init__(
        self,
        domain: Domain,
        n_experiments: int,
        delta: float = 1e-7,
        transform_range: Optional[Bounds] = None,
    ) -> None:
        """Args:
        domain (Domain): A domain defining the DoE domain together with model_type.
        model_type (str or Formula): A formula containing all model terms.
        n_experiments (int): Number of experiments
        delta (float): A regularization parameter for the information matrix. Default value is 1e-7.
        transform_range (Bounds, optional): range to which the input variables are transformed before applying the objective function. Default is None.

        """
        self.domain = deepcopy(domain)

        if transform_range is None:
            self.transform = IndentityTransform()
        else:
            self.transform = MinMaxTransform(
                inputs=self.domain.inputs,
                feature_range=tuple(transform_range),  # type: ignore
            )

        self.n_experiments = n_experiments
        self.delta = delta

        self.vars = self.domain.inputs.get_keys()
        self.n_vars = len(self.domain.inputs)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> float:
        return self._evaluate(self.transform(x=x))

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate_jacobian(self.transform(x=x)) * self.transform.jacobian(
            x=x
        )

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        return (
            self.transform.jacobian(x=x)[None, :]
            * self._evaluate_hessian(self.transform(x=x))
            * self.transform.jacobian(x=x)[:, None]
        )

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def _evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        pass
