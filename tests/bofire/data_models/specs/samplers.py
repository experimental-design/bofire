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
    lambda: {"scramble": True, "optimization": None},
)

specs.add_valid(
    samplers.LatinHypercubeSampler,
    lambda: {"scramble": False, "optimization": "lloyd"},
)

specs.add_invalid(
    samplers.LatinHypercubeSampler,
    lambda: {"optimization": "kmeans"},
    error=ValueError,
    message="Input should be 'random-cd' or 'lloyd'",
)
