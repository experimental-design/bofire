from abc import abstractmethod
from typing import Annotated, Callable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from pydantic import Field, PositiveFloat
from scipy.stats import norm, uniform

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.utils.torch_tools import tkwargs


class OutlierPrior(BaseModel):
    type: str


class UniformOutlierPrior(OutlierPrior):
    type: Literal["UniformOutlierPrior"] = "UniformOutlierPrior"
    bounds: Tuple[float, float]

    def sample(self, n_samples: int) -> np.ndarray:
        return uniform(
            self.bounds[0],
            self.bounds[1] - self.bounds[0],
        ).rvs(n_samples)


class NormalOutlierPrior(OutlierPrior):
    type: Literal["NormalOutlierPrior"] = "NormalOutlierPrior"
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
        return self._domain  # ty: ignore[unresolved-attribute]


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


class FormulationWrapper(Benchmark):
    """Wrapper that turns a benchmark into a formulation benchmark by adding
    spurious features that are not used in the evaluation of the benchmark and
    introducing a sum constraint on all features. The original features get new bounds
    [0, 1/n_original_features] while the spurious features get bounds [0, 1]. On
    evaluation the original features are rescaled back to their original bounds and
    the original benchmark is evaluated. if `n_features_per_original_feature` is larger than 1,
    multiple features per original feature are created and a linear inequality constraint
    is added to ensure that their sum does not exceed 1/n_original_features.

    Via the `max_count` parameter an additional NChooseK constraint can be added to
    the formulation, that limits the number of non-zero non-filler features to `max_count`.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        n_filler_features: int = 1,
        n_features_per_original_feature: int = 1,
        max_count: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._benchmark = benchmark
        benchmark_inputs = self._benchmark.domain.inputs.get()
        assert n_filler_features >= 1, "n_filler_features must be >= 1."
        assert len(benchmark.domain.constraints) == 0, "Constraints not supported yet."
        if not Inputs.is_continuous(benchmark_inputs):
            raise ValueError("Only continuous inputs supported yet.")
        self.n_filler_features = n_filler_features
        self.n_features_per_original_feature = n_features_per_original_feature

        features = []
        constraints = []
        for j, feat in enumerate(benchmark_inputs):
            features += [
                ContinuousInput(
                    key=f"{feat.key}_{i}",
                    bounds=(0, 1 / len(benchmark_inputs)),
                )
                if self.n_features_per_original_feature == 1
                else ContinuousDescriptorInput(
                    key=f"{feat.key}_{i}",
                    bounds=(0, 1 / len(self._benchmark.domain.inputs)),
                    descriptors=self._benchmark.domain.inputs.get_keys(),
                    values=[1 if k == j else 0 for k in range(len(benchmark_inputs))],
                )
                for i in range(self.n_features_per_original_feature)
            ]
            if self.n_features_per_original_feature > 1:
                constraints.append(
                    LinearInequalityConstraint(
                        features=[
                            f"{feat.key}_{i}"
                            for i in range(self.n_features_per_original_feature)
                        ],
                        coefficients=[1.0] * self.n_features_per_original_feature,
                        rhs=1 / len(benchmark_inputs),
                    )
                )

        features += [
            ContinuousInput(key=f"x_filler_{i}", bounds=(0, 1))
            for i in range(self.n_filler_features)
        ]

        inputs = Inputs(features=features)
        constraints.append(
            LinearEqualityConstraint(
                features=inputs.get_keys(),
                coefficients=[1.0] * len(inputs),
                rhs=1.0,
            )
        )
        if max_count is not None:
            constraints.append(
                NChooseKConstraint(
                    features=[
                        key
                        for key in inputs.get_keys()
                        if not key.startswith("x_filler_")
                    ],
                    max_count=max_count,
                    min_count=0,
                    none_also_valid=True,
                )
            )

        self._domain = Domain(
            inputs=inputs,
            constraints=Constraints(constraints=constraints),
            outputs=self._benchmark.domain.outputs,
        )

        self._mins = np.array([feat.bounds[0] for feat in benchmark_inputs])
        self._scales = np.array(
            [feat.bounds[1] - feat.bounds[0] for feat in benchmark_inputs]
        )
        self._scales_new = np.array(
            [1 / len(benchmark_inputs.get_keys())] * len(benchmark_inputs.get_keys())
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for key in self._benchmark.domain.inputs.get_keys():
            X[key] = X[
                [f"{key}_{i}" for i in range(self.n_features_per_original_feature)]
            ].sum(axis=1)

        # drop original columns, only keep latent ones
        X = X.drop(columns=self.domain.inputs.get_keys())
        X = X / self._scales_new

        return self._mins + self._scales * X

    def _f(self, candidates: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X_transformed = self._transform(candidates)
        return self._benchmark._f(X_transformed, **kwargs)

    def get_optima(self) -> pd.DataFrame:
        raise NotImplementedError("Optima not available for FormulationWrapper.")


class SpuriousFeaturesWrapper(Benchmark):
    """Wrapper that adds spurious features to a benchmark, that are ignored on evaluation."""

    def __init__(self, benchmark: Benchmark, n_spurious_features: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert n_spurious_features >= 1, "n_spurious_features must be >= 1."
        self._benchmark = benchmark
        self._domain = Domain(
            inputs=Inputs(
                features=benchmark.domain.inputs.features  # ty: ignore[unsupported-operator]
                + [
                    ContinuousInput(key=f"x_spurious_{i}", bounds=(0, 1))
                    for i in range(n_spurious_features)
                ]
            ),
            outputs=self._benchmark.domain.outputs,
            constraints=self._benchmark.domain.constraints,
        )

    def _f(self, candidates: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._benchmark._f(
            candidates[self._benchmark.domain.inputs.get_keys()], **kwargs
        )

    def get_optima(self) -> pd.DataFrame:
        raise NotImplementedError("Optima not available for SpuriousFeaturesWrapper.")


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
                    ContinuousInput(key=f"x_{i + 1}", bounds=b)
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

    def _f(self, candidates: pd.DataFrame, **kwargs) -> pd.DataFrame:
        Xt = torch.from_numpy(candidates[self.domain.inputs.get_keys()].values).to(
            **tkwargs
        )
        # botorch is very picky regarding the candidates being exactly within the bounds
        # and does not tolerate any numerical noise here. For this reason we clamp the
        # values here.
        Xt = torch.clamp(
            Xt,
            min=torch.tensor(
                [b[0] for b in self.test_function._bounds],
                **tkwargs,
            ),
            max=torch.tensor(
                [b[1] for b in self.test_function._bounds],
                **tkwargs,
            ),
        )
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
