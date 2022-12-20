from typing import Type

from botorch.acquisition import get_acquisition_function
from pydantic import BaseModel, PositiveFloat, validate_arguments, validator

from bofire.domain.constraints import Constraint
from bofire.domain.features import Feature
from bofire.domain.objectives import Objective
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)
from bofire.utils.enum import AcquisitionFunctionEnum


class AcquisitionFunction(BaseModel):
    @staticmethod
    @validate_arguments
    def from_enum(acquistion_function_enum: AcquisitionFunctionEnum):
        if acquistion_function_enum == AcquisitionFunctionEnum.QEI:
            return qEI()
        if acquistion_function_enum == AcquisitionFunctionEnum.QNEI:
            return qNEI()
        if acquistion_function_enum == AcquisitionFunctionEnum.QPI:
            return qPI()
        if acquistion_function_enum == AcquisitionFunctionEnum.QSR:
            return qSR()
        if acquistion_function_enum == AcquisitionFunctionEnum.QUCB:
            return qUCB()
        else:
            raise ValueError(acquistion_function_enum)


class qNEI(AcquisitionFunction):
    pass


class qEI(AcquisitionFunction):
    pass


class qSR(AcquisitionFunction):
    pass


class qUCB(AcquisitionFunction):
    beta: PositiveFloat = 0.2


class qPI(AcquisitionFunction):
    tau: PositiveFloat = 1e-3


class BoTorchSoboStrategy(BotorchBasicBoStrategy):

    acquisition_function: AcquisitionFunction

    @validator("acquisition_function", pre=True)
    def validate_acquisition_function(cls, v):
        if isinstance(v, AcquisitionFunction):
            return v
        else:
            return AcquisitionFunction.from_enum(v)

    def _init_acqf(self) -> None:
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
            self.acquisition_function.__class__.__name__,
            self.model,  # type: ignore
            self.objective,  # type: ignore
            X_observed=X_train,
            X_pending=X_pending,
            constraints=None,
            mc_samples=self.num_sobol_samples,
            qmc=True,
            beta=self.acquisition_function.beta
            if isinstance(self.acquisition_function, qUCB)
            else 0.2,
            tau=self.acquisition_function.tau
            if isinstance(self.acquisition_function, qPI)
            else 1e-3,
        )
        return

    def _init_objective(self):
        self.objective = MultiplicativeObjective(
            targets=[
                var.objective  # type: ignore
                for var in self.domain.outputs().get_by_objective(excludes=None)
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
                for var in self.domain.outputs().get_by_objective(excludes=None)
            ]
        )
        return


class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):
    pass
