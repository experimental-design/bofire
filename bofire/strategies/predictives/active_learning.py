from typing import List

from botorch.acquisition.acquisition import AcquisitionFunction

from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy as DataModel,
)
from bofire.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function

    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        # TODO: Init of the ACQF goes here
        raise NotImplementedError
