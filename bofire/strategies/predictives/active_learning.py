from typing import List, Optional

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, get_acquisition_function
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.acquisition_functions.api import qNegIntPosVar
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.strategies.api import ActiveLearningStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import (
    get_output_constraints,
    tkwargs,
)


class ActiveLearningStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        X_train, X_pending = self.get_acqf_input_tensors()

        (
            objective_callable,
            constraint_callables,
            etas,
        ) = self._get_objective_and_constraints()

        assert self.model is not None

        acqf = get_acquisition_function(
            acquisition_function_name=self.acquisition_function.__class__.__name__,
            model=self.model,
            objective=objective_callable,  # TODO: include posterior transform for active learning mobo
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraint_callables,
            eta=torch.tensor(etas).to(**tkwargs),
            mc_samples=self.num_sobol_samples,
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            prune_baseline=(
                self.acquisition_function.prune_baseline
                if isinstance(self.acquisition_function, (qNegIntPosVar))
                else True
            ),
        )
        return [acqf]
