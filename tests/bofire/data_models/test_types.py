import pytest

from bofire.data_models.base import BaseModel
from bofire.data_models.types import FeatureKeys


def test_FeatureKeys():
    class Bla(BaseModel):
        features: FeatureKeys

    with pytest.raises(ValueError, match="features must be unique"):
        Bla(features=["a", "a"])
