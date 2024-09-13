import numpy as np
import pandas as pd
import pytest

import bofire.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qLogEI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, TaskInput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import QehviStrategy, SoboStrategy
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MultiTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


@pytest.mark.parametrize(
    "task_input, surrogate_2",
    [
        (
            TaskInput(
                key="task", categories=["task_1", "task_2"], allowed=[True, False]
            ),
            MultiTaskGPSurrogate,
        ),
        (
            TaskInput(
                key="task", categories=["task_1", "task_2"], allowed=[False, True]
            ),
            MultiTaskGPSurrogate,
        ),
        (
            TaskInput(
                key="task", categories=["task_1", "task_2"], allowed=[True, True]
            ),
            MultiTaskGPSurrogate,
        ),
        (
            TaskInput(
                key="task", categories=["task_1", "task_2"], allowed=[True, False]
            ),
            SingleTaskGPSurrogate,
        ),
    ],
)
def test_qehvi_with_multitask(task_input, surrogate_2):

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
        }
    )

    input_features = [
        ContinuousInput(key="x", bounds=(0, 1)),
        task_input,
    ]

    objective = MaximizeObjective(w=1)

    inputs = Inputs(features=input_features)

    output_features_1 = [ContinuousOutput(key="y1", objective=objective)]
    output_features_2 = [ContinuousOutput(key="y2", objective=objective)]
    outputs_1 = Outputs(features=output_features_1)
    outputs_2 = Outputs(features=output_features_2)
    outputs = Outputs(features=output_features_1 + output_features_2)

    surrogate_data_1 = MultiTaskGPSurrogate(inputs=inputs, outputs=outputs_1)
    surrogate_data_2 = surrogate_2(inputs=inputs, outputs=outputs_2)
    surrogate_data = [surrogate_data_1, surrogate_data_2]

    # test for error if both models are not multi-task
    if surrogate_2 is SingleTaskGPSurrogate:
        with pytest.raises(ValueError):
            BotorchSurrogates(surrogates=surrogate_data)
        return
    else:
        surrogate_specs = BotorchSurrogates(surrogates=surrogate_data)

    # test for error if task input has more than 1 allowed category
    if sum(task_input.allowed) > 1:
        with pytest.raises(ValueError):
            QehviStrategy(
                domain=Domain(
                    inputs=inputs,
                    outputs=outputs,
                ),
                surrogate_specs=surrogate_specs,
            )
        return
    else:
        strategy_data_model = QehviStrategy(
            domain=Domain(
                inputs=inputs,
                outputs=outputs,
            ),
            surrogate_specs=surrogate_specs,
        )

    strategy = strategies.map(strategy_data_model)
    strategy.tell(experiments)
    candidate = strategy.ask(1)

    # test that the candidate is in the target task
    assert (
        candidate["task"].item()
        == task_input.categories[task_input.allowed.index(True)]
    )


@pytest.mark.parametrize(
    "task_input",
    [
        (TaskInput(key="task", categories=["task_1", "task_2"], allowed=[True, False])),
        (TaskInput(key="task", categories=["task_1", "task_2"], allowed=[False, True])),
        (TaskInput(key="task", categories=["task_1", "task_2"], allowed=[True, True])),
    ],
)
def test_sobo_with_multitask(task_input):
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
            "y": np.concatenate([task_1_y, task_2_y]),
            "task": ["task_1"] * len(task_1_x) + ["task_2"] * len(task_2_x),
        }
    )

    input_features = [
        ContinuousInput(key="x", bounds=(0, 1)),
        task_input,
    ]

    objective = MaximizeObjective(w=1)

    inputs = Inputs(features=input_features)

    output_features = [ContinuousOutput(key="y", objective=objective)]
    outputs = Outputs(features=output_features)

    surrogate_data = [MultiTaskGPSurrogate(inputs=inputs, outputs=outputs)]

    surrogate_specs = BotorchSurrogates(surrogates=surrogate_data)

    # test for error if task input has more than 1 allowed category
    if sum(task_input.allowed) > 1:
        with pytest.raises(ValueError):
            SoboStrategy(
                domain=Domain(
                    inputs=inputs,
                    outputs=outputs,
                ),
                surrogate_specs=surrogate_specs,
            )
        return
    else:
        strategy_data_model = SoboStrategy(
            domain=Domain(
                inputs=inputs,
                outputs=outputs,
            ),
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


test_sobo_with_multitask(
    TaskInput(key="task", categories=["task_1", "task_2"], allowed=[True, False])
)
