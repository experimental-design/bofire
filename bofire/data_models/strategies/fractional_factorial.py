from typing import Any, Literal, Optional, Type

from pydantic import Field, NonNegativeInt, field_validator, model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.strategies.strategy import Strategy
from bofire.utils.doe import get_generator, get_n_blocks, validate_generator


class FractionalFactorialStrategy(Strategy):
    """Two-level (fractional) factorial design, for screening which continuous inputs
    matter with few experiments.

    Every continuous input is set to its lower or upper bound. A full factorial design
    runs all 2^k combinations of k inputs; a fractional one runs a fraction 2^(k-p) of
    them by setting p inputs to products of others, at the price of confounding
    effects with interactions. Center points add runs in the middle of the range. The
    continuous design is repeated for every combination of the categorical and
    discrete inputs.

    Examples:
        >>> FractionalFactorialStrategy(domain=domain, n_generators=1, n_center=2)
    """

    type: Literal["FractionalFactorialStrategy"] = "FractionalFactorialStrategy"
    n_repetitions: NonNegativeInt = Field(
        default=1,
        description="How often the two-level design of the continuous inputs is run; "
        "0 runs only the center points.",
    )
    n_center: NonNegativeInt = Field(
        default=1,
        description="Number of center points per block, at the middle of every "
        "continuous input's range.",
    )
    block_feature_key: Optional[str] = Field(
        default=None,
        description="Key of a categorical or discrete input whose values label the "
        "blocks the design is split into, e.g. days or batches. Its number of values "
        "has to be a possible number of blocks. If not provided, no blocking.",
    )
    generator: Optional[str] = Field(
        default=None,
        description="The design as one term per continuous input, in their order, "
        "e.g. 'a b c ab': single letters are independent factors, a product of "
        "letters sets the input to the product of those factors, and a leading '-' "
        "flips its sign. If not provided, derived from `n_generators`.",
    )
    n_generators: NonNegativeInt = Field(
        default=0,
        description="Number of continuous inputs set to products of the others, p in "
        "the 2^(k-p) runs; 0 is a full factorial design. With a `generator`, it has "
        "to match the generator or be 0.",
    )
    randomize_runorder: bool = Field(
        default=False,
        description="Whether the runs are returned in random order, else in the "
        "standard order of the design.",
    )

    @field_validator("generator", mode="before")
    @classmethod
    def validate_empty_generator(cls, generator: Any) -> Any:
        """Treats the former empty string, meaning no generator, as None."""
        return None if generator == "" else generator

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return False

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousOutput,
            ContinuousInput,
            CategoricalInput,
            DiscreteInput,
        ]

    @model_validator(mode="after")
    def validate_generator(self):
        if self.generator is not None:
            n_products = sum(
                len(term.lstrip("+-")) > 1 for term in self.generator.split()
            )
            if self.n_generators not in (0, n_products):
                raise ValueError(
                    f"n_generators is {self.n_generators}, but the generator sets "
                    f"{n_products} inputs to products of others.",
                )
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
                else len(block_feature.values)  # ty: ignore[possibly-missing-attribute]
            )
            if n_blocks < 2:
                raise ValueError(
                    f"Feature {self.block_feature_key} has only one allowed category/value, blocking is not possible."
                )

            if self.generator is not None:
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
