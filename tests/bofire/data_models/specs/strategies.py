import bofire.data_models.strategies.api as strategies
import bofire.data_models.strategies.predictives.acqf_optimization
from bofire.data_models.acquisition_functions.api import (
    qEI,
    qLogNEHVI,
    qNegIntPosVar,
    qPI,
)
from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalMethodEnum, SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    TaskInput,
)
from bofire.data_models.surrogates.api import BotorchSurrogates, MultiTaskGPSurrogate
from tests.bofire.data_models.specs.api import domain
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])
strategy_commons = {
    "acquisition_optimizer": strategies.BotorchOptimizer(
        **{
            "n_raw_samples": 1024,
            "n_restarts": 8,
            "descriptor_method": CategoricalMethodEnum.EXHAUSTIVE,
            "categorical_method": CategoricalMethodEnum.EXHAUSTIVE,
            "discrete_method": CategoricalMethodEnum.EXHAUSTIVE,
            "maxiter": 2000,
            "batch_limit": 6,
        }
    ),
    "surrogate_specs": BotorchSurrogates(surrogates=[]).model_dump(),
    "outlier_detection_specs": None,
    "seed": 42,
    "min_experiments_before_outlier_check": 1,
    "frequency_check": 1,
    "frequency_hyperopt": 0,
    "folds": 5,
}


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
specs.add_invalid(
    strategies.MoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "acquisition_function": qLogNEHVI().model_dump(),
        **strategy_commons,
        "ref_point": {"o1": 0.5, "o3": 10.0},
    },
    error=ValueError,
    message="Provided refpoint do not match the domain",
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
    strategies.MultiplicativeAdditiveSoboStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        **strategy_commons,
        "acquisition_function": qPI(tau=0.1).model_dump(),
        "use_output_constraints": False,
        "additive_features": ["o1"],
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
    strategies.ActiveLearningStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                    ),
                ],
            ),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        ).model_dump(),
        "acquisition_function": qNegIntPosVar(n_mc_samples=2048).model_dump(),
        **strategy_commons,
    },
)

specs.add_invalid(
    strategies.ActiveLearningStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                    ),
                ],
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="alpha"), ContinuousOutput(key="beta")],
            ),
        ).model_dump(),
        "acquisition_function": qNegIntPosVar(
            n_mc_samples=2048,
            weights={
                "alph_invalid": 0.1,
                "beta_invalid": 0.9,
            },
        ).model_dump(),
        **strategy_commons,
    },
    error=ValueError,
    message="The keys provided for the weights do not match the required keys of the output features.",
)

specs.add_valid(
    strategies.EntingStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "beta": 1.0,
        "bound_coeff": 0.5,
        "acq_sense": "exploration",
        "dist_trafo": "normal",
        "dist_metric": "euclidean_squared",
        "cat_metric": "overlap",
        "num_boost_round": 100,
        "max_depth": 3,
        "min_data_in_leaf": 1,
        "min_data_per_group": 1,
        "verbose": -1,
        "solver_name": "gurobi",
        "solver_verbose": False,
        "solver_params": {},
        "kappa_fantasy": 10.0,
    },
)
specs.add_valid(
    strategies.RandomStrategy,
    lambda: {
        "domain": domain.valid().obj().model_dump(),
        "seed": 42,
        "max_iters": 1000,
        "num_base_samples": 1000,
        "n_burnin": 1000,
        "n_thinning": 32,
        "fallback_sampling_method": SamplingMethodEnum.UNIFORM,
    },
)
for criterion in [
    strategies.AOptimalityCriterion,
    strategies.DOptimalityCriterion,
    strategies.EOptimalityCriterion,
    strategies.GOptimalityCriterion,
    strategies.KOptimalityCriterion,
]:
    for formula in [
        "linear",
        "linear-and-interactions",
        "quadratic",
        "fully-quadratic",
    ]:
        for optimization_strategy in [
            "default",
            "exhaustive",
            "branch-and-bound",
            "partially-random",
            "relaxed",
            "iterative",
        ]:
            for use_cyipopt in [True, False, None]:
                specs.add_valid(
                    strategies.DoEStrategy,
                    lambda criterion=criterion,
                    formula=formula,
                    optimization_strategy=optimization_strategy,
                    use_cyipopt=use_cyipopt: {
                        "domain": domain.valid().obj().model_dump(),
                        "optimization_strategy": optimization_strategy,
                        "verbose": False,
                        "seed": 42,
                        "criterion": criterion(
                            formula=formula, transform_range=None
                        ).model_dump(),
                        "use_hessian": False,
                        "use_cyipopt": use_cyipopt,
                    },
                )

for use_cyipopt in [True, False, None]:
    specs.add_valid(
        strategies.DoEStrategy,
        lambda use_cyipopt=use_cyipopt: {
            "domain": domain.valid().obj().dict(),
            "optimization_strategy": "default",
            "verbose": False,
            "ipopt_options": {"max_iter": 200, "print_level": 0},
            "criterion": strategies.SpaceFillingCriterion(
                sampling_fraction=0.3, transform_range=[-1, 1]
            ).model_dump(),
            "seed": 42,
            "use_hessian": False,
            "use_cyipopt": use_cyipopt,
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
            ).model_dump(),
            strategies.Step(
                strategy_data=strategies.MoboStrategy(
                    domain=tempdomain,
                    acquisition_function=qLogNEHVI(),
                    acquisition_optimizer=strategies.BotorchOptimizer(batch_limit=1),
                ),
                condition=strategies.NumberOfExperimentsCondition(n_experiments=30),
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
                ],
            ),
        ).model_dump(),
        "seed": 42,
    },
)

specs.add_valid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                        local_relative_bounds=(0.2, 0.2),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    ),
                    ContinuousInput(key="c", bounds=(0.1, 0.1)),
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
            constraints=Constraints(
                constraints=[
                    LinearEqualityConstraint(
                        features=["a", "b", "c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    ),
                    LinearInequalityConstraint(
                        features=["a", "b"],
                        coefficients=[1.0, 1.0],
                        rhs=0.95,
                    ),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
        "end": {"a": 0.2, "b": 0.7, "c": 0.1, "d": "b"},
        "atol": 1e-6,
    },
)

specs.add_invalid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                        local_relative_bounds=(0.2, 0.2),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    ),
                    ContinuousInput(key="c", bounds=(0.1, 0.1)),
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
            constraints=Constraints(
                constraints=[
                    LinearEqualityConstraint(
                        features=["a", "b", "c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    ),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"a": 0.8, "b": 0.1, "c": 0.5, "d": "a"},
        "end": {"a": 0.2, "b": 0.7, "c": 0.1, "d": "a"},
    },
    error=ValueError,
    message="`start` is not a valid candidate.",
)

specs.add_invalid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                        local_relative_bounds=(0.2, 0.2),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    ),
                    ContinuousInput(key="c", bounds=(0.1, 0.1)),
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
            constraints=Constraints(
                constraints=[
                    LinearEqualityConstraint(
                        features=["a", "b", "c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    ),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
        "end": {"a": 0.2, "b": 0.9, "c": 0.1, "d": "a"},
    },
    error=ValueError,
    message="`end` is not a valid candidate.",
)


specs.add_invalid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                        local_relative_bounds=(0.2, 0.2),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    ),
                    ContinuousInput(key="c", bounds=(0.1, 0.1)),
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
            constraints=Constraints(
                constraints=[
                    LinearEqualityConstraint(
                        features=["a", "b", "c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    ),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
        "end": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
    },
    error=ValueError,
    message="`start` is equal to `end`.",
)


specs.add_invalid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="a",
                        bounds=(0, 1),
                    ),
                    ContinuousInput(
                        key="b",
                        bounds=(0, 1),
                    ),
                    ContinuousInput(key="c", bounds=(0.1, 0.1)),
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
            constraints=Constraints(
                constraints=[
                    LinearEqualityConstraint(
                        features=["a", "b", "c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    ),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
        "end": {"a": 0.8, "b": 0.1, "c": 0.1, "d": "a"},
    },
    error=ValueError,
    message="Domain has no local search region.",
)

specs.add_invalid(
    strategies.ShortestPathStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key="d", categories=["a", "b", "c"]),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "start": {"d": "a"},
        "end": {"d": "b"},
    },
    error=ValueError,
    message="Domain has no local search region.",
)

specs.add_invalid(
    strategies.SoboStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key=k,
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    )
                    for k in ["a", "b", "c"]
                ],
            ),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
            constraints=Constraints(
                constraints=[
                    NChooseKConstraint(
                        features=["a", "b", "c"],
                        min_count=1,
                        max_count=2,
                        none_also_valid=False,
                    ),
                ],
            ),
        ).model_dump(),
        "acquisition_optimizer": strategies.BotorchOptimizer(
            local_search_config=bofire.data_models.strategies.predictives.acqf_optimization.LSRBO(),
        ),
    },
    error=ValueError,
    message="LSR-BO only supported for linear constraints.",
)

specs.add_invalid(
    strategies.SoboStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key=k,
                        bounds=(0, 1),
                        local_relative_bounds=(0.1, 0.1),
                    )
                    for k in ["a", "b", "c"]
                ]
                + [CategoricalInput(key="d", categories=["a", "b", "c"])],
            ),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
            constraints=Constraints(
                constraints=[InterpointEqualityConstraint(feature="a")],
            ),
        ).model_dump(),
    },
    error=ValueError,
    message="Interpoint constraints can only be used for pure continuous search spaces.",
)


specs.add_valid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "n_repetitions": 1,
        "n_center": 0,
        "n_generators": 0,
        "generator": "",
        "randomize_runorder": False,
    },
)

specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                ],
            ),
        ),
        "block_feature_key": "a",
    },
    error=ValueError,
    message="Feature a not found in discrete/categorical features of domain.",
)

specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    CategoricalInput(
                        key="c",
                        categories=["a", "b", "c"],
                        allowed=[True, False, False],
                    ),
                ],
            ),
        ),
        "block_feature_key": "c",
    },
    error=ValueError,
    message="Feature c has only one allowed category/value, blocking is not possible.",
)

specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    CategoricalInput(
                        key="d",
                        categories=["a", "b", "c"],
                        allowed=[True, True, False],
                    ),
                ],
            ),
        ),
        "block_feature_key": "d",
        "generator": "a b ab",
    },
    error=NotImplementedError,
    message="Blocking is not implemented for custom generators.",
)

specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    CategoricalInput(
                        key="d",
                        categories=["a", "b", "c"],
                        allowed=[True, True, True],
                    ),
                ],
            ),
        ),
        "block_feature_key": "d",
    },
    error=ValueError,
    message="Number of blocks 3 is not possible with 1 repetitions.",
)


specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "n_repetitions": 1,
        "n_center": 0,
        "n_generators": 1,
        "generator": "",
    },
    error=ValueError,
    message="Design not possible, as main factors are confounded with each other.",
)

specs.add_invalid(
    strategies.FractionalFactorialStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                ],
            ),
        ).model_dump(),
        "seed": 42,
        "n_repetitions": 1,
        "n_center": 0,
        "n_generators": 0,
        "generator": "a b c",
    },
    error=ValueError,
    message="Generator does not match the number of factors.",
)

specs.add_invalid(
    strategies.SoboStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    TaskInput(
                        key="task",
                        categories=["task_1", "task_2"],
                        allowed=[True, True],
                    ),
                    ContinuousInput(key="x", bounds=(0, 1)),
                ],
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
        ).model_dump(),
        "surrogate_specs": BotorchSurrogates(
            surrogates=[
                MultiTaskGPSurrogate(
                    inputs=Inputs(
                        features=[
                            TaskInput(
                                key="task",
                                categories=["task_1", "task_2"],
                                allowed=[True, True],
                            ),
                            ContinuousInput(key="x", bounds=(0, 1)),
                        ],
                    ),
                    outputs=Outputs(features=[ContinuousOutput(key="y")]),
                ),
            ],
        ).model_dump(),
    },
    error=ValueError,
    message="Exactly one allowed task category must be specified for strategies with MultiTask models.",
)

specs.add_valid(
    strategies.MultiFidelityStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    TaskInput(
                        key="task", categories=["task_hf", "task_lf"], fidelities=[0, 1]
                    ),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        ).model_dump(),
        **strategy_commons,
        "acquisition_function": qEI().model_dump(),
        "fidelity_thresholds": 0.1,
    },
)

specs.add_invalid(
    strategies.MultiFidelityStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(features=[ContinuousInput(key="a", bounds=(0, 1))]),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        ).model_dump(),
        **strategy_commons,
        "acquisition_function": qEI().model_dump(),
        "fidelity_thresholds": 0.1,
    },
    error=ValueError,
    message="Exactly one task input is required for multi-task GPs.",
)

specs.add_invalid(
    strategies.MultiFidelityStrategy,
    lambda: {
        "domain": Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    TaskInput(
                        key="task", categories=["task_hf", "task_lf"], fidelities=[0, 0]
                    ),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        ).model_dump(),
        **strategy_commons,
        "acquisition_function": qEI().model_dump(),
        "fidelity_thresholds": 0.1,
    },
    error=ValueError,
    message="Only one task can be the target fidelity",
)
