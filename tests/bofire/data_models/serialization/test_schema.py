import pytest

from bofire.data_models.api import AnyThing
from bofire.data_models.base import BaseModel


@pytest.mark.parametrize("model", AnyThing)
def test_domain_models_should_generate_schema(model: BaseModel):
    model.model_json_schema()
