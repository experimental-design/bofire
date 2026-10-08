from typing import Literal

from bofire.data_models.kernels.api import InfiniteWidthBNNKernel
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate


class SingleTaskIBNNSurrogate(SingleTaskGPSurrogate):
    type: Literal["SingleTaskIBNNSurrogate"] = "SingleTaskIBNNSurrogate"
    kernel: InfiniteWidthBNNKernel = InfiniteWidthBNNKernel()
