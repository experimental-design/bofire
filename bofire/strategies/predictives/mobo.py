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

        X_train, X_pending = self.get_acqf_input_tensors()

        # get etas and constraints
        constraints, etas = get_output_constraints(self.domain.outputs)
        if len(constraints) == 0:
            constraints, etas = None, 1e-3
        else:
            etas = torch.tensor(etas).to(**tkwargs)

        objective = self._get_objective()
        # in case that qehvi, qlogehvi is used we need also y
        if isinstance(self.acquisition_function, (qLogEHVI, qEHVI)):
            Y = torch.from_numpy(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    self.experiments
                )[self.domain.outputs.get_keys()].values
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
            mc_samples=self.num_sobol_samples,
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            alpha=self.acquisition_function.alpha,
            prune_baseline=self.acquisition_function.prune_baseline
            if isinstance(self.acquisition_function, (qLogNEHVI, qNEHVI))
            else True,
            Y=Y,
        )
        return [acqf]

    # def _get_acqfs(
    #     self, n
    # ) -> List[
    #     Union[
    #         qExpectedHypervolumeImprovement,
    #         qNoisyExpectedHypervolumeImprovement,
    #         qLogNoisyExpectedHypervolumeImprovement,
    #         qLogExpectedHypervolumeImprovement,
    #     ]
    # ]:
    #     df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
    #         self.experiments
    #     )

    #     train_obj = (
    #         df[self.domain.outputs.get_keys_by_objective(excludes=None)].values
    #         * self.ref_point_mask
    #     )
    #     ref_point = self.get_adjusted_refpoint()
    #     weights = np.array(
    #         [
    #             feat.objective.w  # type: ignore
    #             for feat in self.domain.outputs.get_by_objective(excludes=None)
    #         ]
    #     )
    #     # compute points that are better than the known reference point
    #     better_than_ref = (train_obj > ref_point).all(axis=-1)
    #     # partition non-dominated space into disjoint rectangles
    #     partitioning = NondominatedPartitioning(
    #         ref_point=torch.from_numpy(ref_point * weights),
    #         # use observations that are better than the specified reference point and feasible
    #         Y=torch.from_numpy(train_obj[better_than_ref]),
    #     )

    #     _, X_pending = self.get_acqf_input_tensors()

    #     assert self.model is not None
    #     # setup the acqf
    #     acqf = qExpectedHypervolumeImprovement(
    #         model=self.model,
    #         ref_point=ref_point,  # use known reference point
    #         partitioning=partitioning,
    #         # sampler=self.sampler,
    #         # define an objective that specifies which outcomes are the objectives
    #         objective=self._get_objective(),
    #         X_pending=X_pending,
    #     )
    #     acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
    #     return [acqf]

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
