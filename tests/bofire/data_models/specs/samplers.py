import bofire.data_models.samplers.api as samplers
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(samplers.UniformSampler, lambda: {})

specs.add_valid(
    samplers.SobolSampler,
    lambda: {"scramble": True, "optimization": None},
)

specs.add_valid(
    samplers.SobolSampler,
    lambda: {"scramble": False, "optimization": "random-cd"},
)

specs.add_valid(
    samplers.LatinHypercubeSampler,
    lambda: {"scramble": True, "optimization": None, "strength": 1},
)

specs.add_valid(
    samplers.LatinHypercubeSampler,
    lambda: {"scramble": False, "optimization": "lloyd", "strength": 2},
)

specs.add_invalid(
    samplers.LatinHypercubeSampler,
    lambda: {"strength": 3},
    error=ValueError,
    message="Input should be 1 or 2",
)
