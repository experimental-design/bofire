from typing import Annotated, Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.strategies.strategy import Strategy
from bofire.utils.doe import get_generator, validate_generator


class FractionalFactorialStrategy(Strategy):
    """Fractional factorial design strategy.

    This strategy generates a fractional factorial two level design for the continuous
    part of the domain, which is then combined with the categorical part of the domain.
    For every categorical combination, the continuous part of the design is repeated.
    """

    type: Literal["FractionalFactorialStrategy"] = "FractionalFactorialStrategy"
    n_repetitions: Annotated[
        int,
        Field(
            description="Number of repetitions of the continuous part of the design",
            ge=0,
        ),
    ] = 1
    n_center: Annotated[
        int,
        Field(
            description="Number of center points in the continuous part of the design",
            ge=0,
        ),
    ] = 1
    generator: Annotated[
        str, Field(description="Generator for the continuous part of the design.")
    ] = ""
    n_generators: Annotated[
        int,
        Field(description="Number of reducing factors", ge=0),
    ] = 0
    randomize_runorder: bool = Field(
        default=False,
        description="If true, the run order is randomized, else it is deterministic.",
    )

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return False

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousOutput,
            ContinuousInput,
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            CategoricalMolecularInput,
        ]

    @model_validator(mode="after")
    def validate(self):
        if len(self.generator) > 0:
            validate_generator(len(self.domain.inputs), self.generator)
        else:
            get_generator(
                n_factors=len(self.domain.inputs),
                n_generators=self.n_generators,
            )
        return self
