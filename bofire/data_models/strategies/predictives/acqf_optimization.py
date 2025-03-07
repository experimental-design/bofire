from abc import abstractmethod
from typing import Optional, Literal, Annotated
import warnings

from pydantic import BaseModel, Field
from pydantic import Field, PositiveInt, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo

class AcquisitionOptimizer(BaseModel):
    pass


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
    num_restarts: PositiveInt = 8
    num_raw_samples: IntPowerOfTwo = 1024
    maxiter: PositiveInt = 2000
    batch_limit: Optional[PositiveInt] = Field(default=None, validate_default=True)

    # local search region params
    local_search_config: Optional[AnyLocalSearchConfig] = None

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["num_restarts"],
            info.data["num_restarts"],
        )
        return batch_limit
