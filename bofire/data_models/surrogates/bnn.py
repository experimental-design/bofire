from typing import Literal, Optional

from bofire.data_models.kernels.api import InfiniteWidthBNNKernel
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.trainable import Hyperconfig


class SingleTaskIBNNSurrogate(SingleTaskGPSurrogate):
    type: Literal["SingleTaskIBNNSurrogate"] = "SingleTaskIBNNSurrogate"
    kernel: InfiniteWidthBNNKernel = InfiniteWidthBNNKernel()
    hyperconfig: Optional[Hyperconfig] = None
