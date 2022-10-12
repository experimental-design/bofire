from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from everest.domain.constraints import ConcurrencyConstraint, Constraint
from everest.domain.desirability_functions import (
    IdentityDesirabilityFunction, MaxIdentityDesirabilityFunction,
    MinIdentityDesirabilityFunction)
from everest.domain.features import (ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     InputFeature, OutputFeature)
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.base import BotorchBasicBoStrategy
from everest.strategies.strategy import Strategy
from everest.utils.multiobjective import get_ref_point_mask
from pydantic import validator
from pydantic.types import conint, conlist

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective, WeightedMCMultiOutputObjective)
from botorch.utils.multi_objective.box_decompositions.non_dominated import \
    NondominatedPartitioning


class BoTorchQehviStrategy(BotorchBasicBoStrategy):
    ref_point: Optional[dict]
    acqf: Optional[AcquisitionFunction]
    ref_point_mask: Optional[np.ndarray]
    objective: Optional[MCMultiOutputObjective]


    def _init_acqf(self) -> None:
        df = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        
        train_obj = (
            df[self.domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])].values
            * self.ref_point_mask
        )
        ref_point = self.get_adjusted_refpoint()
        weights = np.array(
            [
                feat.desirability_function.w
                for feat in self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])
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
        # setup the acqf
        self.acqf = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref_point,  # use known reference point
            partitioning=partitioning,
            sampler=self.sampler,
            # define an objective that specifies which outcomes are the objectives
            objective=self.objective,
            # TODO: implement constraints
            # specify that the constraint is on the last outcome
            # constraints=[lambda Z: Z[..., -1]],
        )
        return

    def _init_objective(self) -> None:
        weights = np.array(
            [
                feat.desirability_function.w
                for feat in self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])
            ]
        )
        weights = weights * self.ref_point_mask
        self.objective = WeightedMCMultiOutputObjective(
            outcomes=list(range(len(weights))), weights=torch.from_numpy(weights).to(**tkwargs)
        )
        return

    def _init_domain(self) -> None:
        if len(self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])) < 2:
            raise ValueError(
                "At least two output features has to be defined in the domain."
            )
        for feat in self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]):
            if (
                isinstance(feat.desirability_function, IdentityDesirabilityFunction)
                == False
            ):
                raise ValueError(
                    "Only `MaxIdentityDesirabilityFunction` and `MinIdentityDesirabilityFunction` supported."
                )
            if feat.desirability_function.w != 1.0:
                raise ValueError("Only desirability functions with weight 1 are supported.")
        if self.ref_point is not None:
            if len(self.ref_point) != len(self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])):
                raise ValueError("Dimensionality of provided ref_point does not match number of output features.")
            for feat in self.domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]):
                assert feat in self.ref_point.keys(), f'No reference point defined for output feature {feat}.'
        self.ref_point_mask = get_ref_point_mask(self.domain)
        super()._init_domain()
        return

    def get_adjusted_refpoint(self):
        if self.ref_point is not None:
            return (self.ref_point_mask * np.array([self.ref_point[feat] for feat in self.domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])])).tolist()
        # we have to push all results through the desirability functions and then take the min values
        df = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        return(df[self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])].values*self.ref_point_mask).min(axis=0).tolist()

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == ConcurrencyConstraint:
            return False
        return True


class BoTorchQnehviStrategy(BoTorchQehviStrategy):
    def _init_acqf(self) -> None:
        # TODO move this into general helper function as done in OU
        df = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)

        df = df.drop_duplicates(
            subset=[var.key for var in self.domain.get_features(InputFeature)],
            keep="first",
            inplace=False,
        )
        # now transform it
        df_transform = self.transformer.transform(df)
        # now transform to torch
        train_x = torch.from_numpy(df_transform[self.input_feature_keys].values).to(**tkwargs)
        # if the reference point is not defined it has to be calculated from data
        self.acqf = qNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.get_adjusted_refpoint(),
            X_baseline=train_x,
            sampler=self.sampler,
            prune_baseline=True,
            objective=self.objective,
        )
        return
