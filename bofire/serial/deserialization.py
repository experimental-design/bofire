from typing import Dict

from pydantic import parse_obj_as

from bofire.any.api import (
    AnyConstraint,
    AnyConstraints,
    AnyDomain,
    AnyFeature,
    AnyFeatures,
    AnyKernel,
    AnyModel,
    AnyObjective,
    AnyPrior,
    AnySampler,
    AnyStrategy,
    AnyThing,
)


class Deserialization:
    """Create model instance from serialized data."""

    @staticmethod
    def any(data: Dict) -> AnyThing:
        """Create instance of any model."""

        return parse_obj_as(AnyThing, data)

    @staticmethod
    def constraint(data: Dict) -> AnyConstraint:
        """Create instance of a constraint."""

        return parse_obj_as(AnyConstraint, data)

    @staticmethod
    def constraints(data: Dict) -> AnyConstraints:
        """Create instance of constraints."""

        return parse_obj_as(AnyConstraints, data)

    @staticmethod
    def domain(data: Dict) -> AnyDomain:
        """Create instance of a domain."""

        return parse_obj_as(AnyDomain, data)

    @staticmethod
    def feature(data: Dict) -> AnyFeature:
        """Create instance of a feature."""

        return parse_obj_as(AnyFeature, data)

    @staticmethod
    def features(data: Dict) -> AnyFeatures:
        """Create instance of input or output features."""

        return parse_obj_as(AnyFeatures, data)

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

    @staticmethod
    def sampler(data: Dict) -> AnySampler:
        """Create instance of a sampler."""
        return parse_obj_as(AnySampler, data)
