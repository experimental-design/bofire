from typing import List, Optional

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, get_acquisition_function
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.acquisition_functions.api import (
    qEHVI,
    qLogEHVI,
    qLogNEHVI,
    qNEHVI,
)
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.strategies.api import MoboStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.multiobjective import get_ref_point_mask, infer_ref_point
from bofire.utils.torch_tools import (
    get_multiobjective_objective,
    get_output_constraints,
    tkwargs,
)


class MoboStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.ref_point = data_model.ref_point
        self.ref_point_mask = get_ref_point_mask(self.domain)
        self.acquisition_function = data_model.acquisition_function

    ref_point: Optional[dict] = None
    objective: Optional[MCMultiOutputObjective] = None

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."
        assert self.experiments is not None, "No experiments available."

        X_train, X_pending = self.get_acqf_input_tensors()

        # get etas and constraints
        constraints, etas = get_output_constraints(
            self.domain.outputs,
            experiments=self.experiments,
        )
        if len(constraints) == 0:
            constraints, etas = None, 1e-3
        else:
            etas = torch.tensor(etas).to(**tkwargs)

        objective = self._get_objective()
        # in case that qehvi, qlogehvi is used we need also y
        if isinstance(self.acquisition_function, (qLogEHVI, qEHVI)):
            Y = torch.from_numpy(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    self.experiments,
                )[self.domain.outputs.get_keys()].values,
            ).to(**tkwargs)
        else:
            Y = None

        assert self.model is not None

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,
            ref_point=self.get_adjusted_refpoint(),
            objective=objective,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraints,
            eta=etas,
            mc_samples=self.acquisition_function.n_mc_samples,
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            alpha=self.acquisition_function.alpha,
            prune_baseline=(
                self.acquisition_function.prune_baseline
                if isinstance(self.acquisition_function, (qLogNEHVI, qNEHVI))
                else True
            ),
            Y=Y,
        )
        return [acqf]

    def _get_objective(self) -> GenericMCMultiOutputObjective:
        assert self.experiments is not None
        objective = get_multiobjective_objective(
            outputs=self.domain.outputs,
            experiments=self.experiments,
        )
        return GenericMCMultiOutputObjective(objective=objective)

    def get_adjusted_refpoint(self) -> List[float]:
        assert self.experiments is not None, "No experiments available."
        if self.ref_point is None:
            df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
                self.experiments,
            )
            ref_point = infer_ref_point(
                self.domain,
                experiments=df,
                return_masked=False,
            )
        else:
            ref_point = self.ref_point
        return (
            self.ref_point_mask
            * np.array(
                [
                    ref_point[feat]
                    for feat in self.domain.outputs.get_keys_by_objective(
                        excludes=ConstrainedObjective,
                    )
                ],
            )
        ).tolist()
