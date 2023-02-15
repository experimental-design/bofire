from typing import Dict, Union

from pydantic import parse_obj_as

from bofire.any.constraint import AnyConstraint
from bofire.any.domain import AnyDomain
from bofire.any.feature import AnyFeature
from bofire.any.kernel import AnyKernel
from bofire.any.model import AnyModel
from bofire.any.objective import AnyObjective
from bofire.any.prior import AnyPrior
from bofire.any.strategy import AnyStrategy
from bofire.domain.constraints import Constraints
from bofire.domain.features import InputFeatures, OutputFeatures

Any = Union[
    AnyConstraint,
    AnyDomain,
    AnyFeature,
    AnyModel,
    AnyObjective,
    AnyStrategy,
    AnyKernel,
    AnyPrior,
    Constraints,
    InputFeatures,
    OutputFeatures,
]

# TODO: add AnyConstraints
# TODO: add AnyFeatures


class Deserialization:
    """Create model instance from serialized data."""

    @staticmethod
    def any(data: Dict) -> Any:
        """Create instance of any model."""

        return parse_obj_as(Any, data)

    @staticmethod
    def constraint(data: Dict) -> AnyConstraint:
        """Create instance of a constraint."""

        return parse_obj_as(AnyConstraint, data)

    @staticmethod
    def constraints(data: Dict) -> Constraints:
        """Create instance of constraints."""

        return parse_obj_as(Constraints, data)

    @staticmethod
    def domain(data: Dict) -> AnyDomain:
        """Create instance of a domain."""

        return parse_obj_as(AnyDomain, data)

    @staticmethod
    def feature(data: Dict) -> AnyFeature:
        """Create instance of a feature."""

        return parse_obj_as(AnyFeature, data)

    @staticmethod
    def features(data: Dict) -> Union[InputFeatures, OutputFeatures]:
        """Create instance of input or output features."""

        return parse_obj_as(Union[InputFeatures, OutputFeatures], data)

    @staticmethod
    def kernel(data: Dict) -> AnyKernel:
        """Create instance of a kernel."""

        return parse_obj_as(AnyKernel, data)

    @staticmethod
    def model(data: Dict) -> AnyModel:
        """Create instance of a model."""

        return parse_obj_as(AnyModel, data)

    @staticmethod
    def objective(data: Dict) -> AnyObjective:
        """Create instance of an objective."""

        return parse_obj_as(AnyObjective, data)

    @staticmethod
    def prior(data: Dict) -> AnyPrior:
        """Create instance of an prior."""

        return parse_obj_as(AnyPrior, data)

    @staticmethod
    def strategy(data: Dict) -> AnyStrategy:
        """Create instance of a strategy."""

        return parse_obj_as(AnyStrategy, data)
