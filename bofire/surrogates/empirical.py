import base64
import io
import warnings
from typing import Optional

import torch
from botorch.models.deterministic import DeterministicModel
from typing_extensions import Self

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.surrogates.api import EmpiricalSurrogate as DataModel
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate


class EmpiricalSurrogate(BotorchSurrogate):
    """All necessary functions has to be implemented in the model which can then be loaded
    from cloud pickle.

    Attributes:
        model (DeterministicModel): Botorch model instance.

    """

    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create an empirical runtime surrogate using validated settings.

        Args:
            inputs: Input features.
            outputs: Output features.
            dump: Serialized model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.

        Defaults mirror the data model and are copied for each call. Without a dump, supply
        the empirical model separately before predicting.
        """
        return make_surrogate(cls, DataModel, locals())

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
