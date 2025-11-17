import base64
import io
import warnings
from typing import Optional, cast
from typing_extensions import Self

from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.model_utils import make_surrogate
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

    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        input_preprocessing_specs: InputTransformSpecs = {},
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {}
    ) -> Self:
        """
        Factory method to create an EmpiricalSurrogate from a data model.
        Args:
            # document parameters
        Returns:
            EmpiricalSurrogate: A new instance.
        """
        return cast(Self, make_surrogate(cls, DataModel, locals()))

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
