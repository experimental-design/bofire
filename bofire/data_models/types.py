from typing import Annotated, Dict, List, Union

from pydantic import AfterValidator, Field, PositiveInt

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.molfeatures.api import AnyMolFeatures


def make_unique_validator(name: str):
    """
    Creates a unique validator function for a given name.

    Args:
        name (str): The name of the validator.

    Returns:
        function: The unique validator function.

    Example:
        >>> validator = make_unique_validator("email")
        >>> validator(["john@example.com", "jane@example.com"])
        ['john@example.com', 'jane@example.com']
        >>> validator(["john@example.com", "john@example.com"])
        ValueError: email must be unique
    """

    def validate_unique(uniques: List[str]) -> List[str]:
        """
        Validates that the given list of strings is unique.

        Args:
            uniques (List[str]): The list of strings to validate.

        Returns:
            List[str]: The validated list of strings.

        Raises:
            ValueError: If the strings are not unique.
        """
        if len(uniques) != len(set(uniques)):
            raise ValueError(f"{name} must be unique")
        return uniques

    return validate_unique


def validate_power_of_two(value: int):
    def is_power_of_two(n):
        return (n != 0) and (n & (n - 1) == 0)

    if not is_power_of_two(value):
        raise ValueError("Argument is not power of two.")
    return value


FeatureKeys = Annotated[
    List[str], Field(min_length=2), AfterValidator(make_unique_validator("Features"))
]

CategoryVals = Annotated[
    List[str], Field(min_length=2), AfterValidator(make_unique_validator("Categories"))
]

Descriptors = Annotated[
    List[str], Field(min_length=1), AfterValidator(make_unique_validator("Descriptors"))
]

DiscreteVals = Annotated[List[float], Field(min_length=1)]

InputTransformSpecs = Dict[str, Union[CategoricalEncodingEnum, AnyMolFeatures]]

IntPowerOfTwo = Annotated[PositiveInt, AfterValidator(validate_power_of_two)]
