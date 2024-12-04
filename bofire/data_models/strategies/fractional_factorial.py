from typing import Annotated, Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, Feature
from bofire.data_models.strategies.strategy import Strategy
from bofire.utils.doe import get_generator, validate_generator


class FractionalFactorialStrategy(Strategy):
    type: Literal["FractionalFactorialStrategy"] = "FractionalFactorialStrategy"
    n_repetitions: Annotated[int, Field(description="Number of repetitions", ge=0)] = 1
    n_center: Annotated[int, Field(description="Number of center points", ge=0)] = 1
    generator: Annotated[str, Field(description="Generator for the design.")] = ""
    n_generators: Annotated[
        int,
        Field(description="Number of reducing factors", ge=0),
    ] = 0

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return False

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [ContinuousOutput, ContinuousInput]

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
