import torch
from botorch.models.deterministic import AffineDeterministicModel

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.surrogates.api import (
    CategoricalDeterministicSurrogate as CategoricalDeterministicSurrogateDataModel,
)
from bofire.data_models.surrogates.api import (
    LinearDeterministicSurrogate as LinearDeterministicSurrogateDataModel,
)
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.utils.torch_tools import get_NumericToCategorical_input_transform, tkwargs


class LinearDeterministicSurrogate(BotorchSurrogate):
    def __init__(
        self,
        data_model: LinearDeterministicSurrogateDataModel,
        **kwargs,
    ):
        self.intercept = data_model.intercept
        self.coefficients = data_model.coefficients
        super().__init__(data_model=data_model, **kwargs)
        self.model = AffineDeterministicModel(
            b=data_model.intercept,
            a=torch.tensor(
                [data_model.coefficients[key] for key in self.inputs.get_keys()],
            )
            .to(**tkwargs)
            .unsqueeze(-1),
        )


class CategoricalDeterministicSurrogate(BotorchSurrogate):
    def __init__(
        self,
        data_model: CategoricalDeterministicSurrogateDataModel,
        **kwargs,
    ):
        self.mapping = data_model.mapping
        super().__init__(data_model=data_model, **kwargs)
        self.model = AffineDeterministicModel(
            b=0.0,
            a=torch.tensor(
                [data_model.mapping[key] for key in self.inputs[0].categories],
            )
            .to(**tkwargs)
            .unsqueeze(-1),
        )
        # as bofire always assumes ordinal encoding for categoricals, we have to map them here explicitly
        # to one hot with a specific input transform
        self.model.input_transform = get_NumericToCategorical_input_transform(  # type: ignore
            inputs=self.inputs,
            transform_specs={self.inputs[0].key: CategoricalEncodingEnum.ONE_HOT},
        )
