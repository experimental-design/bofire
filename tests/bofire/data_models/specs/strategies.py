import bofire.data_models.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qPI
from bofire.data_models.enum import CategoricalMethodEnum
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
