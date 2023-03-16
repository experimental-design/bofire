from typing import Literal, Sequence

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum


class MLPEnsemble(BotorchSurrogate):
    type: Literal["MLPEnsemble"] = "MLPEnsemble"
    n_estimators: int
    hidden_layer_sizes: Sequence = (100,)
    activation: Literal["relu", "logistic", "tanh"] = "relu"
    dropout: float = 0.0
    batch_size: int = 10
    n_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    subsample_fraction: float = 1.0
    shuffle: bool = True
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
