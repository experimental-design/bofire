from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionObjective  # type: ignore

from bofire.domain.objectives import Objective


class AquisitionObjective(MCAcquisitionObjective):
    def __init__(self, targets):
        super().__init__()
        self._targets = targets
        return

    @property
    def targets(self):
        return self._targets

    @property
    def num_targets(self):
        return len(self.targets)

    @abstractmethod
    def reward(x: np.ndarray, objective: Objective) -> Tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def forward(self, samples, X=None):
        pass


class MultiplicativeObjective(AquisitionObjective):
    def __init__(self, targets):
        super().__init__(targets)
        return

    def reward(self, x, objective):
        if torch.is_tensor(x):
            reward = torch.sign(objective(x)) * torch.abs(objective(x)) ** objective.w
        else:
            reward = np.sign(objective(x)) * np.abs(objective(x)) ** objective.w

        return reward, 0.0

    def forward(self, samples, X=None):
        infeasible_cost = 0.0
        reward = 1.0

        for i in range(self.num_targets):
            r, infeasible = self.reward(samples[..., i], self.targets[i])
            reward *= r
            infeasible_cost += infeasible  # TODO: where do we check the feasibility of the samples? #TODO:
        return reward - infeasible_cost


class AdditiveObjective(AquisitionObjective):
    def __init__(self, targets):
        super().__init__(targets)
        return

    def reward(self, x, objective):
        return objective(x) * objective.w, 0

    def forward(self, samples, X=None):

        infeasible_cost = 0.0
        reward = 0.0

        for i in range(self.num_targets):
            r, infeasible = self.reward(samples[..., i], self.targets[i])
            reward += r
            infeasible_cost += infeasible
        return reward - infeasible_cost
