import base64
import io
import warnings
from typing import Optional

import torch
from botorch.models.deterministic import DeterministicModel

from bofire.data_models.surrogates.api import EmpiricalSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate


class EmpiricalSurrogate(BotorchSurrogate):
    """All necessary functions has to be implemented in the model which can then be loaded
    from cloud pickle.

    Attributes:
        model (DeterministicModel): Botorch model instance.

    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[DeterministicModel] = None

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from bofire.surrogates import cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO()
        torch.save(self.model, buffer, pickle_module=cloudpickle_module)
        return base64.b64encode(buffer.getvalue()).decode()
        # return codecs.encode(pickle.dumps(self.model), "base64").decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from bofire.surrogates import cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, pickle_module=cloudpickle_module)
