from typing import List

import torch
from botorch.acquisition import qNegIntegratedPosteriorVariance
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedPosteriorTransform

import bofire.strategies.api as strategies
from bofire.data_models.strategies.api import RandomStrategy
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy as DataModel,
)
from bofire.strategies.predictives.botorch import BotorchStrategy


class ActiveLearningStrategy(BotorchStrategy):
    """ActiveLearningStrategy that uses an acquisition function which focuses on
    pure exploration of the objective function only. Can be used for single and multi-objective functions.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self.weights = data_model.weights

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.model is not None

        # sample mc points for integration with the RandomStrategy
        random_model = RandomStrategy(domain=self.domain)
        sampler = strategies.map(random_model)
        mc_points = sampler.ask(candidate_count=self.num_sobol_samples)
        mc_points = torch.tensor(mc_points.values)

        _, X_pending = self.get_acqf_input_tensors()

        ny = len(self.domain.outputs)  # number of outputs
        if ny > 1:
            # create a posterior transform for multi-output models
            if self.weights == None:
                # set all weights equally if nothing is specified
                weights = (
                    torch.ones(ny, dtype=torch.float64) / ny
                )  # normalize weights so they add up to one
            else:
                weights = torch.tensor(self.weights, dtype=torch.float64)
            posterior_transform = ScalarizedPosteriorTransform(weights=weights)
        else:
            posterior_transform = None

        # Instantiate active_learning acquisition function
        acqf = qNegIntegratedPosteriorVariance(
            model=self.model,
            mc_points=mc_points,
            X_pending=X_pending,
            posterior_transform=posterior_transform,
        )
        return [acqf]
