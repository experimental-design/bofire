import pytest

from bofire.data_models.base import BaseModel
from bofire.data_models.types import CategoryVals, FeatureKeys, make_unique_validator


def test_make_unique_validator():
    validator1 = make_unique_validator("Features")
    validator2 = make_unique_validator("Categories")
    with pytest.raises(ValueError, match="Features must be unique"):
        validator1(["a", "a"])
    with pytest.raises(ValueError, match="Categories must be unique"):
        validator2(["a", "a"])


def test_FeatureKeys():
    class Bla(BaseModel):
        features: FeatureKeys
        categories: CategoryVals

    with pytest.raises(ValueError, match="Features must be unique"):
        Bla(features=["a", "a"])

    with pytest.raises(ValueError, match="Categories must be unique"):
        Bla(features=["a", "b"], categories=["a", "a"])
