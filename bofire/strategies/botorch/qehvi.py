from typing import Optional, Type

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from bofire.domain.constraints import Constraint, NChooseKConstraint
from bofire.domain.features import Feature, InputFeature
from bofire.domain.objectives import (
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import tkwargs


# TODO: unite this by using get_acquisiton
class BoTorchQehviStrategy(BotorchBasicBoStrategy):
    ref_point: Optional[dict]
    ref_point_mask: Optional[np.ndarray]
    objective: Optional[MCMultiOutputObjective]

    def _init_acqf(self) -> None:
        df = self.domain.output_features.preprocess_experiments_all_valid_outputs(
            self.experiments
        )

        train_obj = (
            df[self.domain.output_features.get_keys_by_objective(excludes=None)].values
            * self.ref_point_mask
        )
        ref_point = self.get_adjusted_refpoint()
        weights = np.array(
            [
                feat.objective.w  # type: ignore
                for feat in self.domain.output_features.get_by_objective(excludes=None)
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
            # sampler=self.sampler,
            # define an objective that specifies which outcomes are the objectives
            objective=self.objective,
            # TODO: implement constraints
            # specify that the constraint is on the last outcome
            # constraints=[lambda Z: Z[..., -1]],
        )
        # todo comment in after new botorch deployment
        # self.acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
        return

    def _init_objective(self) -> None:
        weights = np.array(
            [
                feat.objective.w  # type: ignore
                for feat in self.domain.output_features.get_by_objective(excludes=None)
            ]
        )
        weights = weights * self.ref_point_mask
        self.objective = WeightedMCMultiOutputObjective(
            outcomes=list(range(len(weights))),
            weights=torch.from_numpy(weights).to(**tkwargs),
        )
        return

    def _init_domain(self) -> None:
        if len(self.domain.output_features.get_by_objective(excludes=None)) < 2:
            raise ValueError(
                "At least two output features has to be defined in the domain."
            )
        for feat in self.domain.output_features.get_by_objective(excludes=None):
            if isinstance(feat.objective, IdentityObjective) is False:  # type: ignore
                raise ValueError(
                    "Only `MaximizeObjective` and `MinimizeObjective` supported."
                )
            if feat.objective.w != 1.0:  # type: ignore
                raise ValueError("Only objectives with weight 1 are supported.")
        if self.ref_point is not None:
            if len(self.ref_point) != len(
                self.domain.output_features.get_by_objective(excludes=None)
            ):
                raise ValueError(
                    "Dimensionality of provided ref_point does not match number of output features."
                )
            for feat in self.domain.output_features.get_keys_by_objective(
                excludes=None
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
                        for feat in self.domain.output_features.get_keys_by_objective(
                            excludes=None
                        )
                    ]
                )
            ).tolist()
        # we have to push all results through the objective functions and then take the min values
        df = self.domain.output_features.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        return (
            (
                df[
                    self.domain.output_features.get_keys_by_objective(excludes=None)
                ].values
                * self.ref_point_mask
            )
            .min(axis=0)
            .tolist()
        )

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        if my_type == NChooseKConstraint:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise
        """
        if my_type not in [MaximizeObjective, MinimizeObjective]:
            return False
        return True


class BoTorchQnehviStrategy(BoTorchQehviStrategy):
    def _init_acqf(self) -> None:
        # TODO move this into general helper function as done in OU
        df = self.domain.output_features.preprocess_experiments_all_valid_outputs(
            self.experiments
        )

        df = df.drop_duplicates(
            subset=[var.key for var in self.domain.get_features(InputFeature)],
            keep="first",
            inplace=False,
        )
        # now transform it
        df_transform = self.transformer.transform(df)  # type: ignore
        # now transform to torch
        train_x = torch.from_numpy(df_transform[self.input_feature_keys].values).to(
            **tkwargs
        )
        # if the reference point is not defined it has to be calculated from data
        self.acqf = qNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.get_adjusted_refpoint(),
            X_baseline=train_x,
            # sampler=self.sampler,
            prune_baseline=True,
            objective=self.objective,
        )
        # todo comment in after new botorch deployment
        # self.acqf._default_sample_shape = torch.Size([self.num_sobol_samples])
        return
