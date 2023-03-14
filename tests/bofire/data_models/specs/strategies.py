import bofire.data_models.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qPI
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.enum import CategoricalMethodEnum, SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput
from tests.bofire.data_models.specs.api import domain
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])


strategy_commons = {
    "num_raw_samples": 1024,
    "num_sobol_samples": 512,
    "num_restarts": 8,
    "descriptor_method": CategoricalMethodEnum.EXHAUSTIVE,
    "categorical_method": CategoricalMethodEnum.EXHAUSTIVE,
    "discrete_method": CategoricalMethodEnum.EXHAUSTIVE,
    "surrogate_specs": None,
}


specs.add_valid(
    strategies.QehviStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.QnehviStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        **strategy_commons,
        "alpha": 0.4,
    },
)
specs.add_valid(
    strategies.QparegoStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.SoboStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).dict(),
    },
)
specs.add_valid(
    strategies.AdditiveSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        "acquisition_function": qPI(tau=0.1).dict(),
        "use_output_constraints": True,
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.MultiplicativeSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).dict(),
    },
)
specs.add_valid(
    strategies.RandomStrategy,
    lambda: {
        "domain": domain.valid().obj().dict(),
    },
)

specs.add_valid(
    strategies.PolytopeSampler,
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
    strategies.RejectionSampler,
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