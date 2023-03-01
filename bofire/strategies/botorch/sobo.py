from typing import Literal, Type, Union

from botorch.acquisition import get_acquisition_function
from botorch.models.gpytorch import GPyTorchModel
from pydantic import (
    BaseModel,
    PositiveFloat,
    parse_obj_as,
    validate_arguments,
    validator,
)

from bofire.domain.constraints import Constraint
from bofire.domain.feature import Feature
from bofire.domain.objective import Objective
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)
from bofire.utils.enum import AcquisitionFunctionEnum


# TODO: move acquisition functions to separate module
# TODO: remove enum
class AcquisitionFunction(BaseModel):
    type: str

    @staticmethod
    @validate_arguments
    def from_enum(acquistion_function_enum: AcquisitionFunctionEnum):
        print(acquistion_function_enum)
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
    type: Literal["qNEI"] = "qNEI"


class qEI(AcquisitionFunction):
    type: Literal["qEI"] = "qEI"


class qSR(AcquisitionFunction):
    type: Literal["qSR"] = "qSR"


class qUCB(AcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: PositiveFloat = 0.2


class qPI(AcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3


# TODO: move this to bofire.any
AnyAquisitionFunction = Union[
    qNEI,
    qEI,
    qSR,
    qUCB,
    qPI,
]


class BoTorchSoboStrategy(BotorchBasicBoStrategy):

    acquisition_function: AcquisitionFunction

    @validator("acquisition_function", pre=True)
    def validate_acquisition_function(cls, v):
        if isinstance(v, AcquisitionFunction):
            return v
        elif isinstance(v, dict):
            return parse_obj_as(AnyAquisitionFunction, v)
        else:
            return AcquisitionFunction.from_enum(v)

    def _init_acqf(self) -> None:
        assert self.is_fitted is True, "Model not trained."

        X_train, X_pending = self.get_acqf_input_tensors()

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
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
        )
        return

    def _init_objective(self):
        self.objective = MultiplicativeObjective(
            targets=[
                var.objective  # type: ignore
                for var in self.domain.outputs.get_by_objective(excludes=None)
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

    type: Literal["BoTorchSoboAdditiveStrategy"] = "BoTorchSoboAdditiveStrategy"

    def _init_objective(self):
        self.objective = AdditiveObjective(
            targets=[
                var.objective  # type: ignore
                for var in self.domain.outputs.get_by_objective(excludes=None)
            ]
        )
        return


class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):

    type: Literal[
        "BoTorchSoboMultiplicativeStrategy"
    ] = "BoTorchSoboMultiplicativeStrategy"
