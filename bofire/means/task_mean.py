import gpytorch
from botorch.models.multitask import _compute_multitask_mean
from torch import Tensor


class TaskConstantMean(gpytorch.means.Mean):
    """A constant mean per task, selected by the task column of the input.

    BoTorch's ``MultiTaskGP`` computes this inside its ``forward``. As a mean module of
    its own, the same computation works in a plain ``SingleTaskGP``.
    """

    def __init__(self, base_mean: gpytorch.means.Mean, num_tasks: int, task_index: int):
        super().__init__()
        self.multitask_mean = gpytorch.means.MultitaskMean(
            base_mean, num_tasks=num_tasks
        )
        self.task_index = task_index

    def forward(self, x: Tensor) -> Tensor:
        t = self.task_index
        return _compute_multitask_mean(
            self.multitask_mean, x[..., :t], x[..., t : t + 1], x[..., t + 1 :]
        )
