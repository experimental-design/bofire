import re
import string
from itertools import combinations
from typing import Annotated, Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, Feature
from bofire.data_models.strategies.strategy import Strategy


class FractionalFactorialStrategy(Strategy):
    type: Literal["FractionalFactorialStrategy"] = "FractionalFactorialStrategy"
    n_repetitions: Annotated[int, Field(description="Number of repetitions", ge=0)] = 1
    n_center: Annotated[int, Field(description="Number of center points", ge=0)] = 1
    generator: Annotated[str, Field(description="Generator for the design.")] = ""
    n_generators: Annotated[
        int, Field(description="Number of reducing factors", ge=0)
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
            self.validate_generator(len(self.domain.inputs), self.generator)
        else:
            self.get_generator(
                n_factors=len(self.domain.inputs), n_generators=self.n_generators
            )
        return self

    @staticmethod
    def validate_generator(n_factors: int, generator: str) -> str:
        if len(generator.split(" ")) != n_factors:
            raise ValueError("Generator does not match the number of factors.")
        # clean it and transform it into a list
        generators = [item for item in re.split(r"\-|\s|\+", generator) if item]
        lengthes = [len(i) for i in generators]

        # Indices of single letters (main factors)
        idx_main = [i for i, item in enumerate(lengthes) if item == 1]

        # Check that single letters (main factors) are unique
        if len([generators[i] for i in idx_main]) != len(
            {generators[i] for i in idx_main}
        ):
            raise ValueError("Main factors are confounded with each other.")

        # Check that single letters (main factors) follow the alphabet
        if (
            "".join(sorted([generators[i] for i in idx_main]))
            != string.ascii_lowercase[: len(idx_main)]
        ):
            raise ValueError("Main factors are not in alphabetical order.")

        # Indices of letter combinations (we need them to fill out H2 properly).
        idx_combi = [i for i, item in enumerate(generators) if item != 1]

        # Check that letter combinations are unique
        if len([generators[i] for i in idx_combi]) != len(
            {generators[i] for i in idx_combi}
        ):
            raise ValueError("Generators are not unique.")

        # Check that only letters are used in the combinations that are also single letters (main factors)
        if not all(
            set(item).issubset({generators[i] for i in idx_main})
            for item in [generators[i] for i in idx_combi]
        ):
            raise ValueError("Generators are not valid.")

        return generator

    @staticmethod
    def get_generator(
        n_factors: int, n_generators: int, seed: Optional[int] = None
    ) -> str:
        if n_generators == 0:
            return " ".join(list(string.ascii_lowercase[:n_factors]))
        n_base_factors = n_factors - n_generators
        if n_generators == 1:
            if n_base_factors == 1:
                raise ValueError(
                    "Design not possible, as main factors are confounded with each other."
                )
            return " ".join(
                list(string.ascii_lowercase[:n_base_factors])
                + [string.ascii_lowercase[:n_base_factors]]
            )
        n_base_factors = n_factors - n_generators
        if n_base_factors - 1 < 2:
            raise ValueError(
                "Design not possible, as main factors are confounded with each other."
            )
        generators = [
            "".join(i)
            for i in (
                combinations(
                    string.ascii_lowercase[:n_base_factors], n_base_factors - 1
                )
            )
        ]
        if len(generators) > n_generators:
            generators = generators[:n_generators]
        elif (n_generators - len(generators) == 1) and (n_base_factors > 1):
            generators += [string.ascii_lowercase[:n_base_factors]]
        elif n_generators - len(generators) >= 1:
            raise ValueError(
                "Design not possible, as main factors are confounded with each other."
            )
        return " ".join(list(string.ascii_lowercase[:n_base_factors]) + generators)
