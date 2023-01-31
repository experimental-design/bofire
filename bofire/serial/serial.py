import json
from typing import Dict, Union

from pydantic import BaseModel, parse_obj_as

from bofire.domain.constraints import Constraints
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.serial.constraints import AnyConstraint
from bofire.serial.domain import AnyDomain
from bofire.serial.features import AnyFeature
from bofire.serial.models import AnyModel
from bofire.serial.objectives import AnyObjective
from bofire.serial.strategies import AnyStrategy

# TODO: simplify imports, remove Any for Singles
Any = Union[
    AnyConstraint,
    AnyDomain,
    AnyFeature,
    AnyModel,
    AnyObjective,
    AnyStrategy,
    Constraints,
]


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
    def model(data: Dict) -> AnyModel:
        """Create instance of a model."""

        return parse_obj_as(AnyModel, data)

    @staticmethod
    def objective(data: Dict) -> AnyObjective:
        """Create instance of an objective."""

        return parse_obj_as(AnyObjective, data)

    @staticmethod
    def strategy(data: Dict) -> AnyStrategy:
        """Create instance of a strategy."""

        return parse_obj_as(AnyStrategy, data)


class Serialization:
    """Create serialized data from model instances."""

    @staticmethod
    def dict(data: BaseModel) -> Dict:
        """Return `data.dict()`."""

        return data.dict()

    @staticmethod
    def json(data: BaseModel) -> str:
        """Return `data.json()`."""

        return data.json()

    @staticmethod
    def json_dict(data: BaseModel) -> Dict:
        """Return `json.loads(data.dict())`."""

        return json.loads(data.json())
