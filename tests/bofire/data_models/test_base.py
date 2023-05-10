import random
import uuid

import pytest
from pydantic.error_wrappers import ValidationError

from bofire.data_models.base import BaseModel


class Bla(BaseModel):
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


def test_forbid_extra():
    with pytest.raises(ValidationError):
        Bla(a=2, mama="papa")
