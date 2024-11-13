import numpy as np
import pandas as pd
import pytest

import bofire.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qLogEI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, TaskInput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import RandomStrategy, SoboStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, MultiTaskGPSurrogate


def _task_1_f(x):
    return np.sin(x * 2 * np.pi)


def _task_2_f(x):
    return 0.9 * np.sin(x * 2 * np.pi) - 0.2 + 0.2 * np.cos(x * 3 * np.pi)


def _domain(task_input):
    input_features = [
        ContinuousInput(key="x", bounds=(0, 1)),
        task_input,
    ]

    objective = MaximizeObjective(w=1)

    inputs = Inputs(features=input_features)

    output_features = [ContinuousOutput(key="y", objective=objective)]
    outputs = Outputs(features=output_features)
    return Domain(inputs=inputs, outputs=outputs)


@pytest.mark.parametrize(
    "task_input",
    [
        (TaskInput(key="task", categories=["task_1", "task_2"], allowed=[False, True])),
        (TaskInput(key="task", categories=["task_1", "task_2"], allowed=[True, False])),
    ],
)
def test_sobo_with_multitask(task_input):
    # set the data

    task_1_x = np.linspace(0.6, 1, 4)
    task_1_y = _task_1_f(task_1_x)

    task_2_x = np.linspace(0, 1, 15)
    task_2_y = _task_2_f(task_2_x)

    experiments = pd.DataFrame(
        {
            "x": np.concatenate([task_1_x, task_2_x]),
            "y": np.concatenate([task_1_y, task_2_y]),
            "task": ["task_1"] * len(task_1_x) + ["task_2"] * len(task_2_x),
        },
    )

    domain = _domain(task_input)
    surrogate_data = [
        MultiTaskGPSurrogate(inputs=domain.inputs, outputs=domain.outputs),
    ]

    surrogate_specs = BotorchSurrogates(surrogates=surrogate_data)  # type: ignore

    strategy_data_model = SoboStrategy(
        domain=domain,
        surrogate_specs=surrogate_specs,
        acquisition_function=qLogEI(),
    )

    strategy = strategies.map(strategy_data_model)
    strategy.tell(experiments)
    candidate = strategy.ask(1)

    # test that the candidate is in the target task
    assert (
        candidate["task"].item()
        == task_input.categories[task_input.allowed.index(True)]
    )


def test_nosurrogate_multitask():
    def test(strat_data_model, **kwargs):
        task_input = TaskInput(
            key="task",
            categories=["task_1", "task_2"],
            allowed=[False, True],
        )
        task_1_x = np.linspace(0.6, 1, 4)
        task_1_y = _task_1_f(task_1_x)
        experiments = pd.DataFrame(
            {
                "x": task_1_x,
                "y": task_1_y,
                "task": ["task_1"] * len(task_1_x),
            },
        )
        domain = _domain(task_input)
        dm = strat_data_model(domain=domain, **kwargs)

        strategy = strategies.map(dm)
        strategy.tell(experiments)
        candidate = strategy.ask(1)
        assert len(candidate) == 1

        task_2_x = np.linspace(0, 1, 15)
        task_2_y = _task_2_f(task_2_x)
        experiments = pd.DataFrame(
            {
                "x": np.concatenate([task_1_x, task_2_x]),
                "y": np.concatenate([task_1_y, task_2_y]),
                "task": ["task_1"] * len(task_1_x) + ["task_2"] * len(task_2_x),
            },
        )
        strategy.tell(experiments)
        candidate = strategy.ask(1)
        assert len(candidate) == 1

    test(RandomStrategy)
    # test(DoEStrategy, formula="linear")
