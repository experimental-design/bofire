from typing import Optional

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from bofire.data_models.objectives.api import BotorchConstrainedObjective
from bofire.data_models.strategies.api import QehviStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import tkwargs


class QehviStrategy(BotorchStrategy):
    """A Bayesian optimization strategy that uses the qExpectedHypervolumeImprovement (qEHVI) acquisition function to maximize the expected hypervolume improvement (EHI).

    The QehviStrategy class is a subclass of the BotorchStrategy class and inherits all its attributes and methods. This class requires a DataModel object as input and optionally accepts any additional keyword arguments that are accepted by the parent class.

    Attributes:
        ref_point (Optional[dict]): A dictionary specifying the reference point for the hypervolume calculation. Defaults to None.
        objective (Optional[MCMultiOutputObjective]): The objective function to optimize. Defaults to None.

    Methods:
        _init_acqf(): Initializes the qEHVI acquisition function.
        _get_objective(): Returns the weighted MCMultiOutputObjective used in the qEHVI acquisition function.
        get_adjusted_refpoint(): Returns the adjusted reference point for the hypervolume calculation.

    See the Botorch documentation for more information on the BotorchStrategy and qEHVI acquisition function.
    """

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

    def _init_acqf(self) -> None:
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
        self.acqf = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref_point,  # use known reference point
            partitioning=partitioning,
            # sampler=self.sampler,
            # define an objective that specifies which outcomes are the objectives
            objective=self._get_objective(),
            X_pending=X_pending,
        )
        self.acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
        return

    def _get_objective(self) -> WeightedMCMultiOutputObjective:
        weights, indices = [], []
        for idx, feat in enumerate(self.domain.outputs.get()):
            if feat.objective is not None and not isinstance(  # type: ignore
                feat.objective, BotorchConstrainedObjective  # type: ignore
            ):
                weights.append(feat.objective.w)  # type: ignore
                indices.append(idx)

        weights = np.array(weights) * self.ref_point_mask
        return WeightedMCMultiOutputObjective(
            outcomes=indices,
            weights=torch.from_numpy(weights).to(**tkwargs),
        )

    def get_adjusted_refpoint(self):
        if self.ref_point is not None:
            return (
                self.ref_point_mask
                * np.array(
                    [
                        self.ref_point[feat]
                        for feat in self.domain.outputs.get_keys_by_objective(
                            excludes=BotorchConstrainedObjective
                        )
                    ]
                )
            ).tolist()
        # we have to push all results through the objective functions and then take the min values
        df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        return (
            (
                df[
                    self.domain.outputs.get_keys_by_objective(
                        excludes=BotorchConstrainedObjective
                    )
                ].values
                * self.ref_point_mask
            )
            .min(axis=0)
            .tolist()
        )
