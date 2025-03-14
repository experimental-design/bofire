from abc import abstractmethod
from typing import Annotated, Literal, Optional, Type

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints import api as constraints
from bofire.data_models.enum import CategoricalMethodEnum
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionOptimizer(BaseModel):
    prefer_exhaustive_search_for_purely_categorical_domains: bool = True

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented. Currently only linear constraints are supported.

        Args:
            my_type (Type[Feature]): The type of the constraint.

        Returns:
            bool: True if the constraint is implemented, False otherwise.

        """
        return True


class LocalSearchConfig(BaseModel):
    """LocalSearchConfigs provide a way to define how to switch between global
    acqf optimization in the global bounds and local acqf optimization in the local
    reference bounds.
    """

    type: str

    @abstractmethod
    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        """Abstract switching function between local and global acqf optimum.

        Args:
            acqf_local (float): Local acqf value.
            acqf_global (float): Global acqf value.

        Returns:
            bool: If true, do local step, else a step towards the global acqf maximum.

        """


class LSRBO(LocalSearchConfig):
    """LSRBO implements the local search region method published in.
    https://www.merl.com/publications/docs/TR2023-057.pdf

    Attributes:
        gamma (float): The switsching parameter between local and global optimization.
            Defaults to 0.1.

    """

    type: Literal["LSRBO"] = "LSRBO"
    gamma: Annotated[float, Field(ge=0)] = 0.1

    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        return acqf_local >= self.gamma


AnyLocalSearchConfig = LSRBO


class BotorchOptimizer(AcquisitionOptimizer):
    n_restarts: PositiveInt = 8
    n_raw_samples: IntPowerOfTwo = 1024
    maxiter: PositiveInt = 2000
    batch_limit: Optional[PositiveInt] = Field(default=None, validate_default=True)

    # encoding params
    descriptor_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    discrete_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE

    # local search region params
    local_search_config: Optional[AnyLocalSearchConfig] = None

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["n_restarts"],
            info.data["n_restarts"],
        )
        return batch_limit

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        if my_type in [
            constraints.NonlinearInequalityConstraint,
            constraints.NonlinearEqualityConstraint,
        ]:
            return False
        return True
