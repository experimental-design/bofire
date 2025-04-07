from itertools import chain

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

# qNoisyExpectedHypervolumeImprovement,
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective

import bofire.data_models.acquisition_functions.api as acquisitions
import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, TaskInput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.predictives.mobo import (
    AbsoluteMovingReferenceValue,
    ExplicitReferencePoint,
    FixedReferenceValue,
    RelativeMovingReferenceValue,
    RelativeToMaxMovingReferenceValue,
)
from bofire.data_models.surrogates.api import BotorchSurrogates, MultiTaskGPSurrogate
from bofire.strategies.api import RandomStrategy
from tests.bofire.utils.test_multiobjective import (
    dfs,
    invalid_domains,
    valid_constrained_domains,
    valid_domains,
)


@pytest.mark.parametrize(
    "domain, ref_point",
    [
        (invalid_domains[0], None),
        (invalid_domains[1], None),
        (valid_domains[0], [0]),
        (valid_domains[0], {}),
        (valid_domains[0], {"of1": 0.0, "of2": 0, "of3": 0}),
        (valid_domains[0], {"of1": 0.0}),
        (valid_domains[0], {"of1": 0.0, "of3": 0.0}),
    ],
)
def test_invalid_mobo(domain, ref_point):
    with pytest.raises(ValueError):
        data_models.MoboStrategy(domain=domain, ref_point=ref_point)


@pytest.mark.parametrize("domain", valid_constrained_domains)
def test_qlognehvi_valid_constrained_objectives(domain):
    data_models.MoboStrategy(domain=domain)


def dict_to_explicit_ref_point_fixed(dict):
    return ExplicitReferencePoint(
        values={k: FixedReferenceValue(value=v) for k, v in dict.items()}
    )


def dict_to_explicit_ref_point_absolutemoving(dict):
    return ExplicitReferencePoint(
        values={
            k: AbsoluteMovingReferenceValue(offset=1, orient_at_best=False)
            for k, v in dict.items()
        }
    )


def dict_to_explicit_ref_point_relativemoving(dict):
    return ExplicitReferencePoint(
        values={k: RelativeMovingReferenceValue(scaling=-0.2) for k, v in dict.items()}
    )


def dict_to_explicit_ref_point_relativetomaxmoving(dict):
    return ExplicitReferencePoint(
        values={
            k: RelativeToMaxMovingReferenceValue(scaling=-0.2) for k, v in dict.items()
        }
    )


@pytest.mark.parametrize(
    "domain, ref_point, experiments, expected",
    [
        (valid_domains[0], {"of1": 0.5, "of2": 10.0}, dfs[0], [0.5, -10.0]),
        (valid_domains[1], {"of1": 0.5, "of3": 0.5}, dfs[1], [0.5, 0.5]),
        (
            valid_domains[0],
            dict_to_explicit_ref_point_fixed({"of1": 0.5, "of2": 10.0}),
            dfs[0],
            [0.5, -10.0],
        ),
        (
            valid_domains[1],
            dict_to_explicit_ref_point_fixed({"of1": 0.5, "of3": 0.5}),
            dfs[1],
            [0.5, 0.5],
        ),
        (
            valid_domains[0],
            dict_to_explicit_ref_point_absolutemoving({"of1": None, "of2": None}),
            dfs[0],
            [2.0, -4.0],
        ),
        (
            valid_domains[1],
            dict_to_explicit_ref_point_absolutemoving({"of1": None, "of3": None}),
            dfs[1],
            [2.0, 3.0],
        ),
        (
            valid_domains[0],
            dict_to_explicit_ref_point_relativemoving({"of1": None, "of2": None}),
            dfs[0],
            [8.2, -2.6],
        ),
        (
            valid_domains[1],
            dict_to_explicit_ref_point_relativemoving({"of1": None, "of3": None}),
            dfs[1],
            [8.2, 4.4],
        ),
        (
            valid_domains[0],
            dict_to_explicit_ref_point_relativetomaxmoving({"of1": None, "of2": None}),
            dfs[0],
            [8.0, -1.6],
        ),
        (
            valid_domains[1],
            dict_to_explicit_ref_point_relativetomaxmoving({"of1": None, "of3": None}),
            dfs[1],
            [8.0, 4.0],
        ),
        (valid_domains[0], None, dfs[0], [1.0, -5.0]),
        (valid_domains[1], None, dfs[1], [1.0, 2.0]),
    ],
)
def test_mobo_get_adjusted_refpoint(domain, ref_point, experiments, expected):
    data_model = data_models.MoboStrategy(domain=domain, ref_point=ref_point)
    strategy = strategies.map(data_model)
    # hack for the test to prevent training of the model when using tell
    strategy.set_experiments(experiments)
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected, np.asarray(adjusted_ref_point))


@pytest.mark.parametrize(
    "strategy, use_ref_point, acqf",
    [
        (data_models.MoboStrategy, use_ref_point, acqf)
        for use_ref_point in [True, False]
        for acqf in [
            acquisitions.qEHVI,
            acquisitions.qLogEHVI,
            acquisitions.qNEHVI,
            acquisitions.qLogNEHVI,
        ]
    ],
)
def test_mobo(strategy, use_ref_point, acqf):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(
        random_strategy.ask(candidate_count=10),
        return_complete=True,
    )
    # init strategy
    data_model = strategy(
        domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None,
        acquisition_function=acqf(),
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)

    bacqf = my_strategy._get_acqfs(2)[0]

    assert isinstance(bacqf.objective, GenericMCMultiOutputObjective)
    if isinstance(acqf, acquisitions.qEHVI):
        assert isinstance(bacqf, qExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qNEHVI):
        assert isinstance(bacqf, qNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogNEHVI):
        assert isinstance(bacqf, qLogNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogEHVI):
        assert isinstance(bacqf, qLogExpectedHypervolumeImprovement)


@pytest.mark.parametrize(
    "acqf",
    [
        acquisitions.qEHVI,
        acquisitions.qLogEHVI,
        acquisitions.qNEHVI,
        acquisitions.qLogNEHVI,
    ],
)
def test_mobo_constraints(acqf):
    benchmark = C2DTLZ2(dim=4)
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)
    data_model = data_models.MoboStrategy(
        domain=benchmark.domain,
        ref_point={"f_0": 1.1, "f_1": 1.1},
        acquisition_function=acqf(),
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)
    bacqf = my_strategy._get_acqfs(2)[0]
    assert isinstance(bacqf.objective, GenericMCMultiOutputObjective)
    if isinstance(acqf, acquisitions.qEHVI):
        assert isinstance(bacqf, qExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qNEHVI):
        assert isinstance(bacqf, qNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogNEHVI):
        assert isinstance(bacqf, qLogNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogEHVI):
        assert isinstance(bacqf, qLogExpectedHypervolumeImprovement)
    assert bacqf.eta == torch.tensor(1e-3)
    assert len(bacqf.constraints) == 1
    assert torch.allclose(
        bacqf.ref_point,
        torch.tensor([-1.1, -1.1], dtype=torch.double),
    )


@pytest.mark.parametrize(
    "num_candidates",
    [1, 2],
)
def test_get_acqf_input(num_candidates):
    # generate data
    num_experiments = 8

    benchmark = DTLZ2(dim=6)
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(
        random_strategy.ask(8),
        return_complete=True,
    )
    data_model = data_models.MoboStrategy(domain=benchmark.domain)
    strategy = strategies.map(data_model)
    # , ref_point=ref_pointw

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.inputs._get_transform_info(
        specs=strategy.surrogate_specs.input_preprocessing_specs,
    )

    assert torch.is_tensor(X_train)
    assert torch.is_tensor(X_pending)
    assert X_train.shape == (
        num_experiments,
        len(set(chain(*names.values()))),
    )
    assert X_pending.shape == (
        num_candidates,
        len(set(chain(*names.values()))),
    )


def test_no_objective():
    domain = DTLZ2(dim=6).domain
    experiments = DTLZ2(dim=6).f(domain.inputs.sample(10), return_complete=True)
    domain.outputs.features.append(ContinuousOutput(key="ignore", objective=None))
    experiments["ignore"] = experiments["f_0"] + 6
    experiments["valid_ignore"] = 1
    data_model = data_models.MoboStrategy(
        domain=domain,
        ref_point={"f_0": 1.1, "f_1": 1.1},
    )
    recommender = strategies.map(data_model=data_model)
    recommender.tell(experiments=experiments)
    candidates = recommender.ask(candidate_count=1)
    recommender.to_candidates(candidates)


@pytest.mark.parametrize(
    "acqf, target_task",
    [
        (acquisitions.qEHVI, "task_1"),
        (acquisitions.qLogEHVI, "task_2"),
        (acquisitions.qNEHVI, "task_1"),
        (acquisitions.qLogNEHVI, "task_2"),
    ],
)
def test_mobo_with_multitask(acqf, target_task):
    # set the data
    def task_1_f(x):
        return np.sin(x * 2 * np.pi)

    def task_2_f(x):
        return 0.9 * np.sin(x * 2 * np.pi) - 0.2 + 0.2 * np.cos(x * 3 * np.pi)

    task_1_x = np.linspace(0.6, 1, 4)
    task_1_y = task_1_f(task_1_x)

    task_2_x = np.linspace(0, 1, 15)
    task_2_y = task_2_f(task_2_x)

    experiments = pd.DataFrame(
        {
            "x": np.concatenate([task_1_x, task_2_x]),
            "y1": np.concatenate([task_1_y, task_2_y]),
            "y2": np.concatenate([task_1_y, task_2_y]),
            "task": ["task_1"] * len(task_1_x) + ["task_2"] * len(task_2_x),
        },
    )

    if target_task == "task_1":
        allowed = [True, False]
    else:
        allowed = [False, True]

    input_features = [
        ContinuousInput(key="x", bounds=(0, 1)),
        TaskInput(key="task", categories=["task_1", "task_2"], allowed=allowed),
    ]

    objective = MaximizeObjective(w=1)

    inputs = Inputs(features=input_features)

    output_features_1 = [ContinuousOutput(key="y1", objective=objective)]
    output_features_2 = [ContinuousOutput(key="y2", objective=objective)]
    outputs_1 = Outputs(features=output_features_1)
    outputs_2 = Outputs(features=output_features_2)
    outputs = Outputs(features=output_features_1 + output_features_2)

    surrogate_data_1 = MultiTaskGPSurrogate(inputs=inputs, outputs=outputs_1)
    surrogate_data_2 = MultiTaskGPSurrogate(inputs=inputs, outputs=outputs_2)
    surrogate_data = [surrogate_data_1, surrogate_data_2]

    surrogate_specs = BotorchSurrogates(surrogates=surrogate_data)

    strategy_data_model = data_models.MoboStrategy(
        domain=Domain(inputs=inputs, outputs=outputs),
        surrogate_specs=surrogate_specs,
        acquisition_function=acqf(),
    )

    strategy = strategies.map(strategy_data_model)
    strategy.tell(experiments)
    candidate = strategy.ask(1)

    # test that the candidate is in the target task
    assert candidate["task"].item() == target_task
