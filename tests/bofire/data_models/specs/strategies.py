import bofire.data_models.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qEI, qLogNEHVI, qPI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalMethodEnum, SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.surrogates.api import BotorchSurrogates
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
    "surrogate_specs": BotorchSurrogates(surrogates=[]).model_dump(),
    "outlier_detection_specs": None,
    "seed": 42,
    "min_experiments_before_outlier_check": 1,
    "frequency_check": 1,
    "frequency_hyperopt": 0,
    "folds": 5,
}


specs.add_valid(
    strategies.QehviStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.QnehviStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        **strategy_commons,
        "alpha": 0.4,
    },
)
specs.add_valid(
    strategies.QparegoStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "acquisition_function": qEI().model_dump(),
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.MoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "acquisition_function": qLogNEHVI().model_dump(),
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.SoboStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(features=[ContinuousInput(key="a", bounds=(0, 1))]),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        ).model_dump(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).model_dump(),
    },
)
specs.add_valid(
    strategies.AdditiveSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "acquisition_function": qPI(tau=0.1).model_dump(),
        "use_output_constraints": True,
        **strategy_commons,
    },
)
specs.add_valid(
    strategies.MultiplicativeSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).model_dump(),
    },
)
specs.add_valid(
    strategies.CustomSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).model_dump(),
        "use_output_constraints": True,
    },
)
specs.add_valid(
    strategies.RandomStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "seed": 42,
    },
)

specs.add_valid(
    strategies.PolytopeSampler,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(0, 1)) for i in range(2)
                ]
            ),
        ).model_dump(),
        "fallback_sampling_method": SamplingMethodEnum.UNIFORM,
        "seed": 42,
        "n_burnin": 1000,
        "n_thinning": 32,
    },
)
specs.add_valid(
    strategies.RejectionSampler,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(0, 1)) for i in range(2)
                ]
            )
        ).model_dump(),
        "max_iters": 1000,
        "num_base_samples": 1000,
        "sampling_method": SamplingMethodEnum.UNIFORM,
        "seed": 42,
    },
)
specs.add_valid(
    strategies.DoEStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "formula": "linear",
        "optimization_strategy": "default",
        "verbose": False,
        "seed": 42,
    },
)
specs.add_valid(
    strategies.UniversalConstraintSampler,
    lambda: {
        "domain": domain.valid().obj().dict(),
        "sampling_fraction": 0.3,
        "ipopt_options": {"maxiter": 200, "disp": 0},
        "seed": 42,
    },
)

tempdomain = domain.valid().obj()

specs.add_valid(
    strategies.StepwiseStrategy,
    lambda: {
        "domain": tempdomain.model_dump(),
        "steps": [
            strategies.Step(
                strategy_data=strategies.RandomStrategy(domain=tempdomain),
                condition=strategies.NumberOfExperimentsCondition(n_experiments=10),
                max_parallelism=2,
            ).model_dump(),
            strategies.Step(
                strategy_data=strategies.QehviStrategy(
                    domain=tempdomain,
                ),
                condition=strategies.NumberOfExperimentsCondition(n_experiments=30),
                max_parallelism=2,
            ).model_dump(),
        ],
        "seed": 42,
    },
)


specs.add_valid(
    strategies.FactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key="alpha", categories=["a", "b", "c"]),
                    DiscreteInput(key="beta", values=[1.0, 2, 3.0, 4.0]),
                ]
            )
        ).model_dump(),
        "seed": 42,
    },
)
