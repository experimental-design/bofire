import bofire.data_models.samplers.api as samplers
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    samplers.PolytopeSampler,
    lambda: {
        "domain": Domain(
            input_features=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", lower_bound=0, upper_bound=1)
                    for i in range(2)
                ]
            ),
        ),
        "fallback_sampling_method": SamplingMethodEnum.UNIFORM,
    },
)
specs.add_valid(
    samplers.RejectionSampler,
    lambda: {
        "domain": Domain(
            input_features=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", lower_bound=0, upper_bound=1)
                    for i in range(2)
                ]
            )
        ),
        "max_iters": 1000,
        "num_base_samples": 1000,
        "sampling_method": SamplingMethodEnum.UNIFORM,
        "num_base_samples": 1000,
        "max_iters": 1000,
    },
)
