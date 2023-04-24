from typing import List, Optional

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.strategies.api import QehviStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.multiobjective import get_ref_point_mask, infer_ref_point
from bofire.utils.torch_tools import get_multiobjective_objective


class QehviStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.ref_point = data_model.ref_point
        self.ref_point_mask = get_ref_point_mask(self.domain)

    ref_point: Optional[dict] = None
    objective: Optional[MCMultiOutputObjective] = None

    def _get_acqfs(self, n) -> List[qExpectedHypervolumeImprovement]:
        df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments
        )

        train_obj = (
            df[self.domain.outputs.get_keys_by_objective(excludes=None)].values
            * self.ref_point_mask
        )
        ref_point = self.get_adjusted_refpoint()
        weights = np.array(
            [
                feat.objective.w  # type: ignore
                for feat in self.domain.outputs.get_by_objective(excludes=None)
            ]
        )
        # compute points that are better than the known reference point
        better_than_ref = (train_obj > ref_point).all(axis=-1)
        # partition non-dominated space into disjoint rectangles
        partitioning = NondominatedPartitioning(
            ref_point=torch.from_numpy(ref_point * weights),
            # use observations that are better than the specified reference point and feasible
            Y=torch.from_numpy(train_obj[better_than_ref]),
        )

        _, X_pending = self.get_acqf_input_tensors()

        assert self.model is not None
        # setup the acqf
        acqf = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref_point,  # use known reference point
            partitioning=partitioning,
            # sampler=self.sampler,
            # define an objective that specifies which outcomes are the objectives
            objective=self._get_objective(),
            X_pending=X_pending,
        )
        acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
        return [acqf]

    def _get_objective(self) -> GenericMCMultiOutputObjective:
        objective = get_multiobjective_objective(outputs=self.domain.outputs)
        return GenericMCMultiOutputObjective(objective=objective)

    def get_adjusted_refpoint(self) -> List[float]:
        if self.ref_point is None:
            df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
                self.experiments
            )
            ref_point = infer_ref_point(
                self.domain, experiments=df, return_masked=False
            )
        else:
            ref_point = self.ref_point
        return (
            self.ref_point_mask
            * np.array(
                [
                    ref_point[feat]
                    for feat in self.domain.outputs.get_keys_by_objective(
                        excludes=ConstrainedObjective
                    )
                ]
            )
        ).tolist()
