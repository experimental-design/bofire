import matplotlib.pyplot as plt
import numpy as np

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, TaskInput
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy as DataModel,
)
from bofire.data_models.surrogates.botorch_surrogates import (
    BotorchSurrogates,
    MultiTaskGPSurrogate,
)
from bofire.strategies.predictives.multi_fidelity import MultiFidelityStrategy


task_input = TaskInput(
    key="task",
    fidelities=[0, 1],
    categories=["task_A", "task_B"],
    allowed=[True, False],
)

domain = Domain.from_lists(
    inputs=[ContinuousInput(key="x", bounds=(0, 1)), task_input],
    outputs=[ContinuousOutput(key="y")],
)

data_model = DataModel(
    domain=domain,
    surrogate_specs=BotorchSurrogates(
        surrogates=[
            MultiTaskGPSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
            )
        ]
    ),
)

strategy = MultiFidelityStrategy(data_model=data_model)


def f(X):
    x = X["x"]
    task_A = np.sin(x * 2 * np.pi)
    task_B = 0.9 * np.sin(x * 2 * np.pi) - 0.2 + 0.2 * np.cos(x * 3 * np.pi)
    return np.where(X["task"] == "task_A", task_A, task_B)


N = 4
seed = 1
np.random.seed(seed)
experiments = domain.inputs.sample(N, seed=seed)
experiments["task"] = np.random.choice(task_input.categories, N)
experiments["y"] = f(experiments)
print(experiments)
strategy.tell(experiments)

for _ in range(20):
    next_experiment = strategy.ask(1)
    next_experiment["y"] = f(next_experiment)
    strategy.tell(next_experiment)

fig, axs = plt.subplots(ncols=2)
axs[0].plot(experiments["y"])
axs[1].hist(experiments["y"])
plt.plot()
