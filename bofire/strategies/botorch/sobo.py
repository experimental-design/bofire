from typing import Type

import torch
from botorch.acquisition import get_acquisition_function

from bofire.domain.constraints import Constraint
from bofire.domain.features import Feature
from bofire.domain.objectives import Objective
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)
from bofire.utils.enum import AcquisitionFunctionEnum
from bofire.utils.torch_tools import tkwargs


class BoTorchSoboStrategy(BotorchBasicBoStrategy):

    acquisition_function: AcquisitionFunctionEnum

    def _init_acqf(
        self, df_pending=None, use_quasi_MC_sampling=True, mc_samples=500, beta=0.2
    ) -> None:
        assert self.is_fitted is True, "Model not trained."

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        transformed = self.transformer.transform(clean_experiments)
        X_train, Y_train = self.get_training_tensors(
            transformed,
            self.domain.output_features.get_keys_by_objective(excludes=None),
        )

        X_pending = None
        if df_pending is not None:
            X_pending = torch.from_numpy(
                self.transformer.transform(df_pending)[self.input_feature_keys].values
            ).to(**tkwargs)

        self.acqf = get_acquisition_function(
            self.acquisition_function.value,
            self.model,
            self.objective,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=None,
            mc_samples=mc_samples,
            qmc=use_quasi_MC_sampling,
            beta=beta,
        )
        return

    def _init_objective(self):
        self.objective = MultiplicativeObjective(
            targets=[
                var.objective
                for var in self.domain.output_features.get_by_objective(excludes=None)
            ]
        )
        return

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
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
        return True


class BoTorchSoboAdditiveStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.additive"

    def _init_objective(self):
        self.objective = AdditiveObjective(
            targets=[
                var.objective
                for var in self.domain.output_features.get_by_objective(excludes=None)
            ]
        )
        return


class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.multiplicative"
