from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import (
    AcquisitionOptimizer as AcquisitionOptimizerDataModel,
)
from bofire.data_models.strategies.api import (
    BotorchOptimizer as BotorchOptimizerDataModel,
)


class AcquisitionOptimizer(ABC):
    def __init__(self, data_model: AcquisitionOptimizerDataModel):
        pass

    @abstractmethod
    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,  # we generate out of the domain all constraints in the format that is needed by the optimizer
        bounds: Tuple[
            List[float], List[float]
        ],  # the bounds are provided by the calling strategy itself and are not
        # generated from the optimizer, this gives the calling strategy the possibility for more control logic
        # as needed for LSRBO or trust region methods
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class BotorchOptimizer(AcquisitionOptimizer):
    def __init__(self, data_model: BotorchOptimizerDataModel):
        self.n_restarts = data_model.n_restarts
        self.n_raw_samples = data_model.n_raw_samples
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit

        # just for completeness here, we should drop the support for FREE and only go over ones that are also
        # allowed, for more speedy optimization we can user other solvers
        # so this can be remomved
        self.categorical_method = data_model.categorical_method
        self.discrete_method = data_model.discrete_method
        self.descriptor_method = data_model.descriptor_method

        self.local_search_config = data_model.local_search_config

        super().__init__(data_model)

    def _setup(self):
        pass

    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        bounds: Tuple[List[float], List[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # this is the implementation of the optimizer, here goes _optimize_acqf_continuous
        pass

OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
}

def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)