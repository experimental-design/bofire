from abc import abstractmethod
from typing import Annotated, Callable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from pydantic import Field, PositiveFloat
from scipy.stats import norm, uniform

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.utils.torch_tools import tkwargs


class OutlierPrior(BaseModel):
    type: str


class UniformOutlierPrior(OutlierPrior):
    type: Literal["UniformOutlierPrior"] = "UniformOutlierPrior"  # type: ignore
    bounds: Tuple[float, float]

    def sample(self, n_samples: int) -> np.ndarray:
        return uniform(
            self.bounds[0],
            self.bounds[1] - self.bounds[0],
        ).rvs(n_samples)


class NormalOutlierPrior(OutlierPrior):
    type: Literal["NormalOutlierPrior"] = "NormalOutlierPrior"  # type: ignore
    loc: float
    scale: PositiveFloat

    def sample(self, n_samples: int) -> np.ndarray:
        return norm(self.loc, self.scale).rvs(n_samples)


AnyOutlierPrior = Union[UniformOutlierPrior, NormalOutlierPrior]


class Benchmark:
    def __init__(
        self,
        outlier_rate: Annotated[float, Field(ge=0, lt=1)] = 0,
        outlier_prior: Optional[AnyOutlierPrior] = None,
    ):
        self.outlier_rate = outlier_rate
        self.outlier_prior = outlier_prior

    def f(
        self,
        candidates: pd.DataFrame,
        return_complete: bool = False,
    ) -> pd.DataFrame:
        Y = self._f(candidates)
        if self.outlier_prior is not None:
            for output_feature in self.domain.outputs.get_keys():
                # no_outliers = int(len(Y) * self.outlier_rate)
                ix2 = np.zeros(len(Y), dtype=bool)
                ix1 = uniform().rvs(len(Y))
                # ix2[np.random.choice(len(Y), no_outliers, replace=False)] = True
                ix2 = ix1 <= self.outlier_rate
                n_outliers = sum(ix2)
                Y.loc[ix2, output_feature] = Y.loc[
                    ix2,
                    output_feature,
                ] + self.outlier_prior.sample(n_outliers)
        if return_complete:
            return pd.concat([candidates, Y], axis=1)

        return Y

    @abstractmethod
    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_optima(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def domain(self) -> Domain:
        return self._domain  # type: ignore


class GenericBenchmark(Benchmark):
    def __init__(
        self,
        domain: Domain,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        outlier_rate: Annotated[float, Field(ge=0, lt=1)] = 0,
        outlier_prior: Optional[AnyOutlierPrior] = None,
    ):
        super().__init__(outlier_prior=outlier_prior, outlier_rate=outlier_rate)
        self._domain = domain
        self.func = func

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return self.func(candidates)


class SyntheticBoTorch(Benchmark):
    """Wrapper around botorch's synthetic test functions.

    Currently supports only continuous single-objective functions.
    """

    def __init__(self, test_function: SyntheticTestFunction, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(
            test_function, SyntheticTestFunction
        ), "Invalid test function."
        # TODO: implement for multi-objective and constrained functions
        if test_function.num_objectives > 1:
            raise NotImplementedError(
                "Multi-objective optimization test functions are not yet supported."
            )
        if hasattr(test_function, "num_constraints"):
            raise NotImplementedError(
                "Multi-objective and constrained optimization test functions are not yet supported."
            )
        # TODO: discrete inds, categorical inds
        # for now we catch all that has categorical or discrete inputs
        if (
            len(test_function.discrete_inds) > 0
            or len(test_function.categorical_inds) > 0
        ):
            raise NotImplementedError(
                "Categorical and discrete inputs are not yet supported."
            )
        self.test_function = test_function
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i+1}", bounds=b)
                    for i, b in enumerate(self.test_function._bounds)
                ]
            ),
            outputs=Outputs(
                features=[
                    ContinuousOutput(
                        key="y",
                        objective=MinimizeObjective(w=1)
                        if self.test_function._is_minimization_by_default
                        else MaximizeObjective(w=1),
                    )
                ]
            ),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        Xt = torch.from_numpy(X[self.domain.inputs.get_keys()].values).to(**tkwargs)
        result = pd.DataFrame(
            self.test_function(Xt).numpy(), columns=self.domain.outputs.get_keys()
        )
        for key in self.domain.outputs.get_keys():
            result[f"valid_{key}"] = 1
        return result

    def get_optima(self) -> pd.DataFrame:
        if self.test_function._optimizers is not None:
            x = pd.DataFrame(
                data=self.test_function._optimizers,
                columns=self.domain.inputs.get_keys(),
            )
            return self.f(x, return_complete=True)
        else:
            raise ValueError("Optima not known for this test function.")
