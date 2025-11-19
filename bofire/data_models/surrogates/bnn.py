from typing import Literal

from bofire.data_models.kernels.api import InfiniteWidthBNNKernel
from bofire.data_models.surrogates.single_task_gp import BaseSingleTaskGPSurrogate


class SingleTaskIBNNSurrogate(BaseSingleTaskGPSurrogate[InfiniteWidthBNNKernel]):
    type: Literal["SingleTaskIBNNSurrogate"] = "SingleTaskIBNNSurrogate"
