from typing import List

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.strategies.api import QnehviStrategy as DataModel
from bofire.strategies.predictives.qehvi import QehviStrategy
from bofire.utils.torch_tools import get_output_constraints, tkwargs


class QnehviStrategy(QehviStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.alpha = data_model.alpha

    def _get_acqfs(self, n) -> List[qNoisyExpectedHypervolumeImprovement]:
        X_train, X_pending = self.get_acqf_input_tensors()

        # get etas and constraints
        constraints, etas = get_output_constraints(self.domain.outputs)
        if len(constraints) == 0:
            constraints, etas = None, 1e-3
        else:
            etas = torch.tensor(etas).to(**tkwargs)

        assert self.model is not None
        # if the reference point is not defined it has to be calculated from data
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.get_adjusted_refpoint(),
            X_baseline=X_train,
            # sampler=self.sampler,
            prune_baseline=True,
            objective=self._get_objective(),
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            X_pending=X_pending,
            constraints=constraints,
            eta=etas,
            alpha=self.alpha,
        )
        acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
        return [acqf]
