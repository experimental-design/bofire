from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from bofire.data_models.domain.api import Inputs


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        pass


class IndentityTransform(Transform):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)


class MinMaxTransform(Transform):
    """This class does the same as sklearn's MinMax Scaler."""

    def __init__(
        self,
        inputs: Inputs,
        n_experiments: int,
        feature_range: Tuple[int, int] = (-1, 1),
    ):
        lower, upper = inputs.get_bounds(specs={})
        self._range = np.tile(np.array(upper) - np.array(lower), n_experiments)
        self._lower = np.array(lower * n_experiments)
        self._transformed_range = feature_range[1] - feature_range[0]
        self._transformed_lower = feature_range[0]
        self._jacobian = self._transformed_range / self._range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (
            x - self._lower
        ) / self._range * self._transformed_range + self._transformed_lower

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._jacobian


AnyTransform = Union[IndentityTransform, MinMaxTransform]
