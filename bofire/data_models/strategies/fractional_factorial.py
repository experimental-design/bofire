from typing import Annotated, Literal, Optional, Type

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
from bofire.utils.doe import get_generator, get_n_blocks, validate_generator


class FractionalFactorialStrategy(Strategy):
    """Fractional factorial design strategy.

    This strategy generates a fractional factorial two level design for the continuous
    part of the domain, which is then combined with the categorical part of the domain.
    For every categorical combination, the continuous part of the design is repeated.
    """

    type: Literal["FractionalFactorialStrategy"] = "FractionalFactorialStrategy"  # type: ignore
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
            description="Number of center points in the continuous part of the design per block",
            ge=0,
        ),
    ] = 1
    block_feature_key: Optional[
        Annotated[
            str,
            Field(
                description="Feature key to use for blocking the design. If not provided, no blocking is used."
            ),
        ]
    ] = None
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
    def validate_generator(self):
        if len(self.generator) > 0:
            validate_generator(
                len(self.domain.inputs.get(ContinuousInput)), self.generator
            )
        else:
            get_generator(
                n_factors=len(self.domain.inputs.get(ContinuousInput)),
                n_generators=self.n_generators,
            )
        return self

    @model_validator(mode="after")
    def validate_blocking(self):
        if self.block_feature_key is not None:
            if self.block_feature_key not in self.domain.inputs.get_keys(
                includes=[DiscreteInput, CategoricalInput]
            ):
                raise ValueError(
                    f"Feature {self.block_feature_key} not found in discrete/categorical features of domain."
                )
            block_feature = self.domain.inputs.get_by_key(self.block_feature_key)
            n_blocks = (
                len(block_feature.get_allowed_categories())
                if isinstance(block_feature, CategoricalInput)
                else len(block_feature.values)  # type: ignore
            )
            if n_blocks < 2:
                raise ValueError(
                    f"Feature {self.block_feature_key} has only one allowed category/value, blocking is not possible."
                )

            if len(self.generator) > 0:
                raise NotImplementedError(
                    "Blocking is not implemented for custom generators."
                )

            n_factors = len(self.domain.inputs.get(ContinuousInput))

            n_possible_blocks = get_n_blocks(
                n_factors=n_factors,
                n_generators=self.n_generators,
                n_repetitions=self.n_repetitions,
            )

            if n_blocks not in n_possible_blocks:
                raise ValueError(
                    f"Number of blocks {n_blocks} is not possible with {self.n_repetitions} repetitions."
                )

        return self
