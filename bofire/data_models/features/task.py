from typing import Any, ClassVar, List, Literal

import numpy as np
from pydantic import Field, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.feature import Input


class TaskInput(Input):
    type: Literal["TaskInput"] = "TaskInput"


class CategoricalTaskInput(TaskInput, CategoricalInput):
    order_id: ClassVar[int] = 8
    type: Literal["CategoricalTaskInput"] = "CategoricalTaskInput"
    fidelities: List[int] = []

    @model_validator(mode="after")
    def validate_fidelities(self):
        n_tasks = len(self.categories)
        if self.fidelities == []:
            for _ in range(n_tasks):
                self.fidelities.append(0)
        if len(self.fidelities) != n_tasks:
            raise ValueError(
                "Length of fidelity lists must be equal to the number of tasks",
            )
        if list(set(self.fidelities)) != list(range(np.max(self.fidelities) + 1)):
            raise ValueError(
                "Fidelities must be a list containing integers, starting from 0 and increasing by 1",
            )
        return self


class CostModel(BaseModel):
    """Function mapping task value to experimental cost."""

    type: Any


class AffineFidelityCostModel(CostModel):
    type: Literal["AffineFidelityCostModel"] = "AffineFidelityCostModel"
    fidelity_weight: float
    fixed_cost: float


class CostAwareUtility(BaseModel):
    """Function mapping the cost of an experiment to its utility."""

    type: Any
    cost_model: CostModel


class InverseCostWeightedUtility(CostAwareUtility):
    type: Literal["InverseCostWeightedUtility"] = "InverseCostWeightedUtility"


class ContinuousTaskInput(TaskInput, ContinuousInput):
    order_id: ClassVar[int] = 11
    type: Literal["ContinuousTaskInput"] = "ContinuousTaskInput"  # type: ignore
    cost_aware_utility: CostAwareUtility = Field(
        default_factory=lambda: InverseCostWeightedUtility(
            cost_model=AffineFidelityCostModel(fidelity_weight=1.0, fixed_cost=0.0)
        )
    )
