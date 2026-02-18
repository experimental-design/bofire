from typing import ClassVar, Literal

import numpy as np
from pydantic import NonNegativeFloat, PositiveFloat, model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.feature import Input


class TaskInput(Input):
    type: Literal["TaskInput"] = "TaskInput"


class CategoricalTaskInput(TaskInput, CategoricalInput):
    order_id: ClassVar[int] = 8
    type: Literal["CategoricalTaskInput"] = "CategoricalTaskInput"
    fidelities: list[int] = []
    fidelity_costs: list[NonNegativeFloat] = []

    @model_validator(mode="after")
    def validate_fidelities(self):
        n_tasks = len(self.categories)
        if self.fidelities == []:
            for _ in range(n_tasks):
                self.fidelities.append(0)
        if len(self.fidelities) != n_tasks:
            raise ValueError(
                "Length of fidelity list must be equal to the number of tasks",
            )
        if list(set(self.fidelities)) != list(range(np.max(self.fidelities) + 1)):
            raise ValueError(
                "Fidelities must be a list containing integers, starting from 0 and increasing by 1",
            )
        return self

    @model_validator(mode="after")
    def validate_fidelity_costs(self):
        n_tasks = len(self.categories)
        if self.fidelity_costs == []:
            for _ in range(n_tasks):
                self.fidelity_costs.append(1.0)
        if len(self.fidelity_costs) != n_tasks:
            raise ValueError(
                "Length of fidelity cost list must be equal to the number of tasks",
            )
        return self


class ContinuousTaskInput(TaskInput, ContinuousInput):
    order_id: ClassVar[int] = 11
    type: Literal["ContinuousTaskInput"] = "ContinuousTaskInput"  # type: ignore
    fidelity_fixed_cost: NonNegativeFloat = 1.0
    fidelity_cost_weight: PositiveFloat = 1.0


class DiscreteTaskInput(TaskInput, DiscreteInput):
    order_id: ClassVar[int] = 12
    type: Literal["DiscreteTaskInput"] = "DiscreteTaskInput"  # type: ignore
    fidelity_fixed_cost: NonNegativeFloat = 1.0
    fidelity_cost_weight: PositiveFloat = 1.0
