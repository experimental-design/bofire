from typing import Literal, Optional

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionFunctionOptimizer(BaseModel):
    """
    Base data model for acquisition function optimizers

    Attributes:
        type (Literal): Type of optimizer.
    """

    type: str


class BotorchAcqfOptimizer(AcquisitionFunctionOptimizer):
    """
    Botorch optimizers for acquisition function optimization

    Attributes:
        type (Literal): `BotorchAcqfOptimizer`.
        num_restarts (PositiveInt): Number of starting points in a multi-start strategy for the continuous subspace of
            the search domain. The initial points are selected from `num_raw_samples` random samples.
        num_raw_samples (IntPowerOfTwo): Number of (pseudo-)random samples for initialization of the multi-start
            strategy. Has to be a power of two.
        maxiter (PositiveInt): Maximum number of iterations for each optimization.
        batch_limit (Optional[PositiveInt], optional): Number of optimizations that are batched together in
            a single call to the optimizer.
    """

    type: Literal["BotorchAcqfOptimizer"] = "BotorchAcqfOptimizer"
    num_restarts: PositiveInt = 8
    num_raw_samples: IntPowerOfTwo = 1024
    maxiter: PositiveInt = 2000
    batch_limit: Optional[PositiveInt] = Field(default=None, validate_default=True)

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["num_restarts"], info.data["num_restarts"]
        )
        return batch_limit


AnyAcqfOptimizer = BotorchAcqfOptimizer
