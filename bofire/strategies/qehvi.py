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
from bofire.strategies.botorch import BotorchStrategy
from bofire.strategies.multiobjective import get_ref_point_mask
from bofire.surrogates.torch_tools import tkwargs


# TODO: unite this by using get_acquisiton
class QehviStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.ref_point = data_model.ref_point
        # self._init_domain()

    ref_point: Optional[dict] = None
    ref_point_mask: Optional[np.ndarray] = None
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
            # TODO: implement constraints
            # specify that the constraint is on the last outcome
            # constraints=[lambda Z: Z[..., -1]],
        )
        # todo comment in after new botorch deployment
        # self.acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
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

    def _init_domain(self) -> None:
        super()._init_domain()
        # TODO: check for None
        if (
            len(
                self.domain.outputs.get_by_objective(
                    excludes=BotorchConstrainedObjective
                )
            )
            < 2
        ):
            raise ValueError(
                "At least two output features has to be defined in the domain."
            )
        for feat in self.domain.outputs.get_by_objective(
            excludes=BotorchConstrainedObjective
        ):
            if feat.objective.w != 1.0:  # type: ignore
                raise ValueError("Only objectives with weight 1 are supported.")
        if self.ref_point is not None:
            if len(self.ref_point) != len(
                self.domain.outputs.get_by_objective(
                    excludes=BotorchConstrainedObjective
                )
            ):
                raise ValueError(
                    "Dimensionality of provided ref_point does not match number of output features."
                )
            for feat in self.domain.outputs.get_keys_by_objective(
                excludes=BotorchConstrainedObjective
            ):
                assert (
                    feat in self.ref_point.keys()
                ), f"No reference point defined for output feature {feat}."
        self.ref_point_mask = get_ref_point_mask(self.domain)
        super()._init_domain()
        return

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
