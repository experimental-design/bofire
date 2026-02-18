from typing import ClassVar, Literal

import numpy as np
from pydantic import model_validator

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


class ContinuousTaskInput(TaskInput, ContinuousInput):
    order_id: ClassVar[int] = 11
    type: Literal["ContinuousTaskInput"] = "ContinuousTaskInput"  # type: ignore


class DiscreteTaskInput(TaskInput, DiscreteInput):
    order_id: ClassVar[int] = 12
    type: Literal["DiscreteTaskInput"] = "DiscreteTaskInput"  # type: ignore
