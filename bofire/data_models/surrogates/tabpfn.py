from typing import Literal, Optional, Type

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TabPFNSurrogate(TrainableBotorchSurrogate):
    type: Literal["TabPFNSurrogate"] = "TabPFNSurrogate"

    tabpfn_version: Literal["v2", "v2.5", "v2.6", "v3"] = "v3"
    posterior_type: Literal["gaussian", "riemann"] = "gaussian"
    # TabPFN's KV cache is incompatible with ``requires_grad`` inputs, and BO
    # acquisition optimization always needs gradients through X. Off by default;
    # opt in only for grad-free prediction workflows.
    use_kv_cache: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    checkpoint_path: Optional[str] = None

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        return issubclass(my_type, ContinuousOutput)
