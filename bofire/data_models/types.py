from typing import Annotated, List

from pydantic import AfterValidator, Field


def validate_features_unique(features: List[str]) -> List[str]:
    """
    Validates that the given list of features is unique.

    Args:
        features (List[str]): The list of features to validate.

    Returns:
        List[str]: The validated list of features.

    Raises:
        ValueError: If the features are not unique.
    """
    if len(features) != len(set(features)):
        raise ValueError("features must be unique")
    return features


FeatureKeys = Annotated[
    List[str], Field(min_length=2), AfterValidator(validate_features_unique)
]
