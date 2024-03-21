import torch
from botorch.models.deterministic import AffineDeterministicModel

from bofire.data_models.surrogates.api import LinearDeterministicSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.utils.torch_tools import tkwargs


class LinearDeterministicSurrogate(BotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.intercept = data_model.intercept
        self.coefficients = data_model.coefficients
        super().__init__(data_model=data_model, **kwargs)
        self.model = AffineDeterministicModel(
            b=data_model.intercept,
            a=torch.tensor(
                [data_model.coefficients[key] for key in self.inputs.get_keys()]
            )
            .to(**tkwargs)
            .unsqueeze(-1),
        )
