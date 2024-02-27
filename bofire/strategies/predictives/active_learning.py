from typing import List

import torch
from botorch.acquisition import qNegIntegratedPosteriorVariance
from botorch.acquisition.acquisition import AcquisitionFunction

from bofire.data_models.strategies.api import RandomStrategy
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy as DataModel,
)
import bofire.strategies.api as strategies
from bofire.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        
    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.model is not None

        # sample mc points for integration with the RandomStrategy
        random_model = RandomStrategy(domain=self.domain)
        sampler = strategies.map(random_model)
        mc_points = sampler.ask(candidate_count=self.num_sobol_samples)
        mc_points = torch.tensor(mc_points.values)

        _, X_pending = self.get_acqf_input_tensors()

        # Instantiate active_learning acquisition function
        acqf = qNegIntegratedPosteriorVariance(
            model=self.model,
            mc_points=mc_points,
            X_pending=X_pending
        )
        return [acqf]
