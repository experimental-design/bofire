import pytest
from pydantic import TypeAdapter, ValidationError

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import CategoricalInput, ContinuousInput
from bofire.data_models.samplers.api import (
    AnySampler,
    LatinHypercubeSampler,
    SobolSampler,
    UniformSampler,
    sampler_from_method,
)
from bofire.data_models.strategies.api import RandomStrategy


INPUTS = Inputs(
    features=[
        ContinuousInput(key="a", bounds=(0, 1)),
        ContinuousInput(key="b", bounds=(-1, 1)),
        CategoricalInput(key="c", categories=["x", "y", "z"]),
    ]
)


@pytest.mark.parametrize(
    "method, sampler_kwargs, sampler",
    [
        (SamplingMethodEnum.UNIFORM, None, UniformSampler()),
        (SamplingMethodEnum.SOBOL, {"scramble": False}, SobolSampler(scramble=False)),
        (SamplingMethodEnum.LHS, {"strength": 2}, LatinHypercubeSampler(strength=2)),
    ],
)
def test_sample_with_method_equals_sampler(method, sampler_kwargs, sampler):
    assert sampler_from_method(method, sampler_kwargs) == sampler
    by_method = INPUTS.sample(9, method, seed=1, sampler_kwargs=sampler_kwargs)
    by_sampler = INPUTS.sample(9, sampler, seed=1)
    assert by_method.equals(by_sampler)
    assert set(by_sampler["c"]) <= {"x", "y", "z"}


def test_sampler_rejects_unknown_option():
    with pytest.raises(ValidationError):
        sampler_from_method(SamplingMethodEnum.SOBOL, {"bits": 30})
    with pytest.raises(ValidationError):
        TypeAdapter(AnySampler).validate_python(
            {"type": "LatinHypercubeSampler", "strength": 3}
        )


def test_random_strategy_loads_former_sampling_fields():
    stored = {
        "type": "RandomStrategy",
        "domain": {"inputs": INPUTS.model_dump()},
        "fallback_sampling_method": "LHS",
        "sampler_kwargs": {"scramble": False},
    }
    strategy = RandomStrategy.model_validate(stored)
    assert strategy.fallback_sampler == LatinHypercubeSampler(scramble=False)
    assert "sampler_kwargs" not in strategy.model_dump()
