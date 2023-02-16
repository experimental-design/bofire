import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Type

from bofire.domain.constraint import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.constraints import Constraints
from bofire.domain.domain import Domain
from bofire.domain.feature import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.domain.objectives import (
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
)
from bofire.models.gps.gps import MixedSingleTaskGPModel, ScalerEnum, SingleTaskGPModel
from bofire.models.gps.kernels import (
    AdditiveKernel,
    HammondDistanceKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.models.gps.priors import (
    GammaPrior,
    NormalPrior,
    botorch_lengthcale_prior,
    botorch_scale_prior,
)


class Spec:
    """Spec <spec> for class <cls>."""

    def __init__(self, cls: Type, spec: dict):
        self.cls = cls
        # TODO: change spec to Union[dict, Callable] and "generate" for each call if callable
        self.spec = spec

    def obj(self, **kwargs) -> Any:
        """Create and return an instance of <cls>."""

        return self.cls(**{**self.spec, **kwargs})

    def typed_spec(self) -> dict:
        """Return the <spec>, extended by {'type': <cls>.__name__}."""
        return {
            **self.spec,
            "type": self.cls.__name__,
        }

    def __str__(self):
        return f"{self.cls.__name__}: {self.spec}"

    def __repr__(self):
        return str(self)


class Invalidator(ABC):
    """Invalidator rule to invalidate a given spec."""

    @abstractmethod
    def invalidate(self, spec: Spec) -> List[Spec]:
        """Return a list of invalidated specs.

        If this invalidator is not applicable to the specified
        spec, an empty list is returned."""

        pass


class Overwrite(Invalidator):
    """Overwrite properties if the key is contained in the spec."""

    def __init__(self, key: str, overwrites: List[dict]):
        self.key = key
        self.overwrites = overwrites

    def invalidate(self, spec: Spec) -> List[Spec]:
        if self.key not in spec.spec:
            return []
        return [
            Spec(spec.cls, {**spec.spec, **overwrite}) for overwrite in self.overwrites
        ]


class Specs:
    """Collection of valid and invalid specs.

    In the init, only <invalidators> must be provided.
    Valid specs are added via the <add_valid> method.
    Invalid specs can auomatically be added as part of this method."""

    def __init__(self, invalidators: List[Invalidator]):
        self.invalidators = invalidators
        self.valids: List[Spec] = []
        self.invalids: List[Spec] = []

    def _get_spec(self, specs: List[Spec], cls: Type = None):
        if cls is not None:
            specs = [s for s in specs if s.cls == cls]
        if len(specs) == 0 and cls is None:
            raise TypeError("no spec found")
        elif len(specs) == 0:
            raise TypeError(f"no spec of type {cls.__name__} found")
        return random.choice(specs)

    def valid(self, cls: Type = None) -> Spec:
        """Return a valid spec.

        If <cls> is provided, the list of all valid specs is filtered by it.
        If no spec (with the specified class) exists, a TypeError is raised.
        If more than one spec exist, a random one is returned."""

        return self._get_spec(self.valids, cls)

    def invalid(self, cls: Type = None) -> Spec:
        """Return an invalid spec.

        If <cls> is provided, the list of all invalid specs is filtered by it.
        If no spec (with the specified class) exists, a TypeError is raised.
        If more than one spec exist, a random one is returned."""

        return self._get_spec(self.invalids, cls)

    def add_valid(self, cls: Type, spec: dict, add_invalids: bool = True) -> Spec:
        """Add a new valid spec to the list.

        If <add_invalids> is True (default), invalid specs are generated using the
        rules provided in <invalidators>."""

        spec_ = Spec(cls, spec)
        self.valids.append(spec_)
        if add_invalids:
            for invalidator in self.invalidators:
                self.invalids += invalidator.invalidate(spec_)
        return spec_


# # # # # # # # # # # # # # # # # #
# objectives
# # # # # # # # # # # # # # # # # #

objectives = Specs(
    [
        Overwrite(
            "w",
            [
                {"w": 0},
                {"w": -100},
                {"w": 1.0000001},
                {"w": 100},
            ],
        ),
        Overwrite(
            "lower_bound",
            [
                {"lower_bound": 5, "upper_bound": 3},
                {"lower_bound": None, "upper_bound": None},
                {"lower_bound": 5, "upper_bound": None},
                {"lower_bound": None, "upper_bound": 3},
            ],
        ),
        Overwrite(
            "steepness",
            [
                {"steepness": 0},
                {"steepness": -100},
            ],
        ),
        Overwrite(
            "tolerance",
            [
                {"tolerance": -0.1},
                {"tolerance": -100},
            ],
        ),
    ]
)

objectives.add_valid(
    CloseToTargetObjective,
    {
        "target_value": 42,
        "exponent": 2,
        "tolerance": 1.5,
        "w": 1.0,
    },
)
objectives.add_valid(
    ConstantObjective,
    {
        "value": 0.2,
        "w": 0.4,
    },
)
objectives.add_valid(
    DeltaObjective,
    {
        "w": 0.3,
        "lower_bound": 0.1,
        "upper_bound": 0.9,
        "ref_point": 1,
        "scale": 2,
    },
)
objectives.add_valid(
    MaximizeObjective,
    {
        "w": 0.3,
        "lower_bound": 0.1,
        "upper_bound": 0.9,
    },
)
objectives.add_valid(
    MaximizeSigmoidObjective,
    {
        "steepness": 0.2,
        "tp": 0.3,
        "w": 0.4,
    },
)
objectives.add_valid(
    MinimizeObjective,
    {
        "w": 0.3,
        "lower_bound": 0.1,
        "upper_bound": 0.9,
    },
)
objectives.add_valid(
    MinimizeSigmoidObjective,
    {
        "steepness": 0.2,
        "tp": 0.3,
        "w": 0.4,
    },
)
objectives.add_valid(
    TargetObjective,
    {
        "w": 0.2,
        "target_value": 0.4,
        "tolerance": 0.4,
        "steepness": 0.3,
    },
)


# # # # # # # # # # # # # # # # # #
# features
# # # # # # # # # # # # # # # # # #

features = Specs(
    [
        # TODO: enable the invalidator
        # Overwrite(
        #     "lower_bound",
        #     [
        #         {"lower_bound": 5, "upper_bound": 3},
        #         {"lower_bound": None, "upper_bound": None},
        #         {"lower_bound": 5, "upper_bound": None},
        #         {"lower_bound": None, "upper_bound": 3},
        #     ],
        # ),
    ]
)

features.add_valid(
    DiscreteInput,
    {
        "key": str(uuid.uuid4()),
        "values": [1.0, 2.0, 2.5],
    },
)
features.add_valid(
    ContinuousInput,
    {
        "key": str(uuid.uuid4()),
        "lower_bound": 3.0,
        "upper_bound": 5.3,
    },
)
features.add_valid(
    ContinuousDescriptorInput,
    {
        "key": str(uuid.uuid4()),
        "lower_bound": 3,
        "upper_bound": 5.3,
        "descriptors": ["d1", "d2"],
        "values": [1.0, 2.0],
    },
)
features.add_valid(
    CategoricalInput,
    {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [True, True, False],
    },
)
features.add_valid(
    CategoricalDescriptorInput,
    {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [True, True, False],
        "descriptors": ["d1", "d2"],
        "values": [
            [1.0, 2.0],
            [3.0, 7.0],
            [5.0, 1.0],
        ],
    },
)
features.add_valid(
    ContinuousOutput,
    {
        "key": str(uuid.uuid4()),
        "objective": objectives.valid(MaximizeObjective).typed_spec(),
    },
)


# # # # # # # # # # # # # # # # # #
# constraints
# # # # # # # # # # # # # # # # # #

constraints = Specs(
    [
        Overwrite(
            "expression",
            [
                {"expression": [1, 2, 3]},
            ],
        ),
        Overwrite(
            "coefficients",
            [
                {"features": [], "coefficients": []},
                {"features": [], "coefficients": [1]},
                {"features": ["f1", "f2"], "coefficients": [-0.4]},
                {"features": ["f1", "f2"], "coefficients": [-0.4, 1.4, 4.3]},
                {"features": ["f1", "f1"], "coefficients": [1, 1]},
                {"features": ["f1", "f1", "f2"], "coefficients": [1, 1, 1]},
            ],
        ),
    ]
)

constraints.add_valid(
    LinearEqualityConstraint,
    {
        "features": ["f1", "f2", "f3"],
        "coefficients": [1, 2, 3],
        "rhs": 1.5,
    },
)
constraints.add_valid(
    LinearInequalityConstraint,
    {
        "features": ["f1", "f2", "f3"],
        "coefficients": [1, 2, 3],
        "rhs": 1.5,
    },
)
constraints.add_valid(
    NonlinearEqualityConstraint,
    {
        "expression": "f1*f2",
    },
)
constraints.add_valid(
    NonlinearInequalityConstraint,
    {
        "expression": "f1*f2",
    },
)
constraints.add_valid(
    NChooseKConstraint,
    {
        "features": ["f1", "f2", "f3"],
        "min_count": 1,
        "max_count": 1,
        "none_also_valid": False,
    },
)


# # # # # # # # # # # # # # # # # #
# models
# # # # # # # # # # # # # # # # # #

models = Specs([])
models.add_valid(
    MixedSingleTaskGPModel,
    {
        "input_features": InputFeatures(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "output_features": OutputFeatures(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "continuous_kernel": MaternKernel(ard=True, nu=2.5),
        "categorical_kernel": HammondDistanceKernel(ard=True),
        "scaler": ScalerEnum.NORMALIZE.value,
        "model": None,
        "input_preprocessing_specs": {},
        "training_specs": {},
    },
)
models.add_valid(
    SingleTaskGPModel,
    {
        "input_features": InputFeatures(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "output_features": OutputFeatures(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=botorch_lengthcale_prior()
            ),
            outputscale_prior=botorch_scale_prior(),
        ),
        "scaler": ScalerEnum.NORMALIZE.value,
        "model": None,
        "input_preprocessing_specs": {},
        "training_specs": {},
    },
)


# # # # # # # # # # # # # # # # # #
# domain
# # # # # # # # # # # # # # # # # #

domains = Specs([])
domains.add_valid(
    Domain,
    {
        "input_features": InputFeatures(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "output_features": OutputFeatures(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "constraints": Constraints(),
        "experiments": None,
        "candidates": None,
    },
)
domains.add_valid(
    Domain,
    {
        "input_features": InputFeatures(),
        "output_features": OutputFeatures(),
        "constraints": Constraints(),
        "experiments": {
            "a": [1, 2, 3, 4],
            "b": [3, 4, 5, 6],
            "c": [10, 2, -4, 5],
        },
        "candidates": {
            "d": [5, 2, 5],
            "e": [3, 4, 5],
        },
    },
)


# # # # # # # # # # # # # # # # # #
# prior
# # # # # # # # # # # # # # # # # #

priors = Specs([])
priors.add_valid(
    NormalPrior,
    {
        "loc": 0.4,
        "scale": 0.9,
    },
)
priors.add_valid(
    GammaPrior,
    {
        "concentration": 0.3,
        "rate": 0.6,
    },
)


# # # # # # # # # # # # # # # # # #
# kernel
# # # # # # # # # # # # # # # # # #

kernels = Specs([])
kernels.add_valid(
    HammondDistanceKernel,
    {
        "ard": False,
    },
)
kernels.add_valid(
    LinearKernel,
    {},
)
kernels.add_valid(
    MaternKernel,
    {
        "ard": True,
        "nu": 0.415,
        "lengthscale_prior": priors.valid().obj(),
    },
)
kernels.add_valid(
    RBFKernel,
    {
        "ard": True,
        "lengthscale_prior": priors.valid().obj(),
    },
)
kernels.add_valid(
    ScaleKernel,
    {
        "base_kernel": kernels.valid(LinearKernel).obj(),
        "outputscale_prior": priors.valid().obj(),
    },
)
kernels.add_valid(
    AdditiveKernel,
    {
        "kernels": [
            kernels.valid(LinearKernel).obj(),
            kernels.valid(MaternKernel).obj(),
        ]
    },
)
kernels.add_valid(
    MultiplicativeKernel,
    {
        "kernels": [
            kernels.valid(LinearKernel).obj(),
            kernels.valid(MaternKernel).obj(),
        ]
    },
)
