import pytest

from bofire.benchmarks.api import MultiTaskHimmelblau
from bofire.data_models.acquisition_functions.api import qMFMES, qMFVariance
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import TaskInput
from bofire.data_models.strategies.api import (
    MultiFidelityStrategy as MultiFidelityStrategyDataModel,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.api import MultiFidelityStrategy, RandomStrategy


def test_mf_requires_all_fidelities_observed():
    benchmark = MultiTaskHimmelblau()
    (task_input,) = benchmark.domain.inputs.get(TaskInput, exact=True)
    assert task_input.type == "TaskInput"

    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(
            domain=benchmark.domain,
            fallback_sampling_method=SamplingMethodEnum.SOBOL,
            seed=42,
        ),
    )

    experiments = benchmark.f(random_strategy.ask(100), return_complete=True)

    domain_with_extra_task = Domain(
        inputs=benchmark.domain.inputs.get(excludes=TaskInput)  # type: ignore
        + (
            TaskInput(
                key=task_input.key,
                categories=["task_1", "task_dummy", "task_2"],
                fidelities=[0, 1, 2],
                allowed=[True, True, True],
            ),
        ),
        outputs=benchmark.domain.outputs,
    )

    strategy = MultiFidelityStrategy(
        data_model=MultiFidelityStrategyDataModel(
            domain=domain_with_extra_task,
            fidelity_acquisition_function=qMFVariance(
                fidelity_thresholds=[0.1, 0.1, 0.1]
            ),
        )
    )

    # since there are no observations on task_dummy, the strategy should raise an error
    # in Python 3.9, a more cryptic RuntimeError is raised by gpytorch
    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"(Some tasks have no experiments)|(index out of row bound)",
    ):
        strategy.tell(experiments)
        strategy.ask(1)

    # test that the strategy does not raise an error if all fidelities are observed
    experiments.loc[experiments.index[-1], task_input.key] = "task_dummy"
    strategy.tell(experiments, replace=True)
    strategy.ask(1)


@pytest.mark.parametrize(
    "fidelity_acqf",
    (
        qMFVariance(fidelity_thresholds=0.1, beta=0.2),
        qMFMES(fidelity_costs=[2.0, 1.0]),
    ),
)
def test_mf_fidelity_selection(fidelity_acqf):
    benchmark = MultiTaskHimmelblau()
    (task_input,) = benchmark.domain.inputs.get(TaskInput, exact=True)
    assert task_input.type == "TaskInput"
    task_input.fidelities = [0, 1]

    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(
            domain=benchmark.domain,
            fallback_sampling_method=SamplingMethodEnum.SOBOL,
            seed=42,
        ),
    )

    N_train = 10
    experiments = benchmark.f(random_strategy.ask(N_train), return_complete=True)
    experiments[task_input.key] = ["task_1"] + ["task_2"] * (N_train - 1)
    experiments, withheld = experiments.iloc[:-1], experiments.iloc[-1:]

    strategy = MultiFidelityStrategy(
        data_model=MultiFidelityStrategyDataModel(
            domain=benchmark.domain,
            fidelity_acquisition_function=fidelity_acqf,
        )
    )

    strategy.tell(experiments)
    # test that for a point close to training data, the highest fidelity is selected
    good_training_point = experiments[benchmark.domain.outputs.get_keys()[0]].argmin()
    close_to_training = experiments.iloc[
        good_training_point : good_training_point + 1
    ].copy()
    close_to_training[benchmark.domain.inputs.get_keys(excludes=TaskInput)] += 0.01
    pred = strategy.select_fidelity_candidate(close_to_training)
    assert (pred[task_input.key] == task_input.categories[0]).all()

    # test that for a point far from training data, the lowest fidelity is selected
    pred = strategy.select_fidelity_candidate(withheld)
    assert (pred[task_input.key] == task_input.categories[1]).all()


@pytest.mark.parametrize(
    "fidelity_acqf",
    (
        qMFVariance(fidelity_thresholds=0.1, beta=0.2),
        qMFMES(fidelity_costs=[2.0, 1.0]),
    ),
)
def test_mf_point_selection(fidelity_acqf):
    benchmark = MultiTaskHimmelblau()
    (task_input,) = benchmark.domain.inputs.get(TaskInput, exact=True)
    assert task_input.type == "TaskInput"
    task_input.fidelities = [0, 1]

    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(
            domain=benchmark.domain,
            fallback_sampling_method=SamplingMethodEnum.SOBOL,
            seed=42,
        ),
    )

    experiments = benchmark.f(random_strategy.ask(4), return_complete=True)
    experiments[task_input.key] = ["task_1", "task_2", "task_2", "task_2"]

    strategy = MultiFidelityStrategy(
        data_model=MultiFidelityStrategyDataModel(
            domain=benchmark.domain,
            fidelity_acquisition_function=fidelity_acqf,
        )
    )

    strategy.tell(experiments)

    # smoke test for querying a new point
    candidate = strategy.ask(1)
    assert set(benchmark.domain.inputs.get_keys()).issubset(candidate.columns)
