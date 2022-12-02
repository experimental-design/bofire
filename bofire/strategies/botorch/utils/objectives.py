from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionObjective

from bofire.domain.objectives import (
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.utils.torch_tools import tkwargs


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

    """TODO: reformulate reward with callable of objective, needs refactoring of objective to return tensors
    @abstractmethod
    def reward(x: np.array, objective: Objective) -> Tuple[torch.tensor, float]:
        pass"""

    @abstractmethod
    def reward_constant(self, x: np.array, value, w) -> torch.tensor:
        pass

    @abstractmethod
    def reward_target(
        self, x: torch.tensor, w, target_value, tolerance, steepness
    ) -> torch.tensor:
        pass

    @abstractmethod
    def reward_close_to_target(
        self, x: torch.tensor, w, target_value, tolerance, exponent
    ) -> torch.tensor:
        pass

    @abstractmethod
    def reward_min(
        self, x: torch.tensor, w: np.array, tp: np.array, steepness: np.array, **kwargs
    ) -> np.array:
        pass

    @abstractmethod
    def reward_max(
        self, x: torch.tensor, w: np.array, tp: np.array, steepness: np.array, **kwargs
    ) -> np.array:
        pass

    @abstractmethod
    def reward_max_identity(
        self, x: torch.tensor, w: np.array, lower_bound: np.array, upper_bound: np.array
    ) -> torch.tensor:
        pass

    @abstractmethod
    def reward_min_identity(
        self, x: torch.tensor, w: np.array, lower_bound: np.array, upper_bound: np.array
    ) -> torch.tensor:
        pass

    @abstractmethod
    def reward_delta_identity(
        x: torch.tensor, w: np.array, ref_point: np.array
    ) -> torch.tensor:
        pass

    @abstractmethod
    def forward(self, samples, X=None):
        pass

    def reward(self, x, objective):
        if isinstance(objective, TargetObjective):
            return self.reward_target(x, **objective.dict()), 0.0
        if isinstance(objective, CloseToTargetObjective):
            return self.reward_close_to_target(x, **objective.dict()), 0.0
        if isinstance(objective, MinimizeSigmoidObjective):
            return self.reward_min(x, **objective.dict()), 0.0
        if isinstance(objective, MaximizeSigmoidObjective):
            return self.reward_max(x, **objective.dict()), 0.0
        if isinstance(objective, MaximizeObjective):
            return self.reward_max_identity(x, **objective.dict()), 0.0
        if isinstance(objective, MinimizeObjective):
            return self.reward_min_identity(x, **objective.dict()), 0.0
        if isinstance(objective, ConstantObjective):
            return self.reward_constant(x, **objective.dict()), 0.0
        if isinstance(objective, DeltaObjective):
            # other option is to use non negative costs --> we first try the identity option
            return self.reward_delta_identity(x, **objective.dict()), 0.0
        # we don't have the case ignore?
        # if desirability_function == "ignore":
        #     return torch.ones(x.shape), 0.
        else:
            raise NotImplementedError


class MultiplicativeObjective(AquisitionObjective):
    def __init__(self, targets):
        super().__init__(targets)
        return

    """TODO: reformulate reward with callable of objective, needs refactoring of objective to return tensors
    def reward(self, x, objective):
        return torch.sign(objective(x)) * torch.abs(objective(x)) ** objective.w, 0.0"""

    def reward_target(self, x, w, target_value, tolerance, steepness):
        return (
            1.0
            / (1.0 + torch.exp(-1 * steepness * (x - (target_value - tolerance))))
            * (
                1.0
                - 1.0
                / (1.0 + torch.exp(-1.0 * steepness * (x - (target_value + tolerance))))
            )
        ) ** w

    def reward_close_to_target(self, x, w, target_value, tolerance, exponent):
        return (torch.abs(x - target_value) ** exponent - tolerance**exponent) ** w

    def reward_min(self, x, w, tp, steepness, **kwargs):
        return ((1.0 - 1.0 / (1.0 + torch.exp(-1.0 * steepness * (x - tp))))) ** w

    def reward_max(self, x, w, tp, steepness, **kwargs):
        return (1.0 / (1.0 + torch.exp(-1.0 * steepness * (x - tp)))) ** w

    def reward_max_identity(self, x, w, lower_bound, upper_bound, **kwargs):
        return ((x - lower_bound) / (upper_bound - lower_bound)) ** w

    def reward_min_identity(self, x, w, lower_bound, upper_bound, **kwargs):
        return -1.0 * ((x - lower_bound) / (upper_bound - lower_bound)) ** w

    def reward_delta_identity(self, x, w, ref_point, scale, **kwargs):
        return ((ref_point - x) * scale) ** w

    def reward_constant(self, x, value, w):
        return (torch.ones(x.shape).to(**tkwargs) * value) ** w

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

    """TODO: reformulate reward with callable of objective, needs refactoring of objective to return tensors
    def reward(self, x, objective):
        return objective(x) * objective.w, 0"""

    def reward_target(self, x, w, target_value, tolerance, steepness):
        return (
            w
            * 1.0
            / (1.0 + torch.exp(-1 * steepness * (x - (target_value - tolerance))))
            * (
                1.0
                - 1.0
                / (1.0 + torch.exp(-1.0 * steepness * (x - (target_value + tolerance))))
            )
        )

    def reward_close_to_target(self, x, w, target_value, tolerance, exponent):
        return (torch.abs(x - target_value) ** exponent - tolerance**exponent) * w

    def reward_min(self, x, w, tp, steepness, **kwargs):
        return w * (1.0 - 1.0 / (1.0 + torch.exp(-1.0 * steepness * (x - tp))))

    def reward_max(self, x, w, tp, steepness, **kwargs):
        return w * 1.0 / (1.0 + torch.exp(-1.0 * steepness * (x - tp)))

    def reward_max_identity(self, x, w, lower_bound, upper_bound, **kwargs):
        return (x - lower_bound) / (upper_bound - lower_bound) * w

    def reward_min_identity(self, x, w, lower_bound, upper_bound, **kwargs):
        return -1.0 * (x - lower_bound) / (upper_bound - lower_bound) * w

    def reward_delta_identity(self, x, scale, w, ref_point, **kwargs):
        return ((ref_point - x) * scale) * w

    def reward_constant(self, x, value, w):
        return torch.ones(x.shape).to(**tkwargs) * value * w

    def forward(self, samples, X=None):

        infeasible_cost = 0.0
        reward = 0.0

        for i in range(self.num_targets):
            r, infeasible = self.reward(samples[..., i], self.targets[i])
            reward += r
            infeasible_cost += infeasible
        return reward - infeasible_cost
