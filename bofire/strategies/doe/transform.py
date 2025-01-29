from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from bofire.data_models.domain.api import Inputs


class Transform(ABC):
    def __init__(*args, **kwargs):
        pass

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
        feature_range: Tuple[float, float] = (-1, 1),
    ):
        lower, upper = inputs.get_bounds(specs={})
        self._range = np.array(upper) - np.array(lower)
        self._lower = lower
        self._transformed_range = feature_range[1] - feature_range[0]
        self._transformed_lower = feature_range[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - np.array(self._lower * (len(x) // len(self._lower)))) / np.tile(
            self._range,
            len(x) // len(self._range),
        ) * self._transformed_range + self._transformed_lower

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._transformed_range / np.tile(
            self._range,
            len(x) // len(self._range),
        )


AnyTransform = Union[IndentityTransform, MinMaxTransform]
