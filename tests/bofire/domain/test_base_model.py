import random
import uuid

import pytest
from pydantic.error_wrappers import ValidationError

from bofire.domain.util import PydanticBaseModel


class Bla(PydanticBaseModel):
    a: int = 1


@pytest.fixture
def bla():
    return Bla()


@pytest.fixture
def a():
    return random.randint(10, 100)


@pytest.fixture
def b():
    return str(uuid.uuid4())


def test_assignment_validation(bla, a):
    bla.a = a
    assert bla.a == a


def test_assignment_validation_invalid(bla, b):
    with pytest.raises(ValidationError):
        bla.a = b
