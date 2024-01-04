import pytest
from pydantic.error_wrappers import ValidationError

from bofire.data_models.base import BaseModel


class Blub(BaseModel):
    b: str = "mama"


class Bla(BaseModel):
    a: Blub


def test_assignment_validation():
    bla = Bla(a=Blub(b="mama"))
    bla.a = Blub(b="papa")
    assert bla.a == Blub(b="papa")


def test_assignment_validation_invalid():
    bla = Bla(a=Blub(b="mama"))
    with pytest.raises(ValidationError):
        bla.a = 42.0


def test_forbid_extra():
    with pytest.raises(ValidationError):
        Bla(a=Blub(b="mama"), b=5)


def test_copy_on_validation():
    blub = Blub(b="papa")
    bla = Bla(a=blub)
    assert id(blub) == id(bla.a)
    blub.b = "lotta"
    assert blub.b == bla.a.b == "lotta"
