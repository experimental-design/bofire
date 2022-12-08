from typing import Type

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


class BoTorchSoboStrategy(BotorchBasicBoStrategy):

    acquisition_function: AcquisitionFunctionEnum

    def _init_acqf(self, beta=0.2) -> None:
        assert self.is_fitted is True, "Model not trained."

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        transformed = self.transformer.transform(clean_experiments)  # type: ignore
        X_train, _ = self.get_training_tensors(
            transformed,
            self.domain.output_features.get_keys_by_objective(excludes=None),  # type: ignore
        )

        # TODO: refactor pending experiments
        X_pending = None

        self.acqf = get_acquisition_function(
            self.acquisition_function.value,
            self.model,  # type: ignore
            self.objective,  # type: ignore
            X_observed=X_train,
            X_pending=X_pending,
            constraints=None,
            mc_samples=self.num_sobol_samples,
            qmc=True,
            beta=beta,
        )
        return

    def _init_objective(self):
        self.objective = MultiplicativeObjective(
            targets=[
                var.objective  # type: ignore
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
    def _init_objective(self):
        self.objective = AdditiveObjective(
            targets=[
                var.objective  # type: ignore
                for var in self.domain.output_features.get_by_objective(excludes=None)
            ]
        )
        return


class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):
    pass
