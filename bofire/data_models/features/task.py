from typing import ClassVar, List, Literal

import numpy as np
from pydantic import model_validator

from bofire.data_models.features.categorical import CategoricalInput


class TaskInput(CategoricalInput):
    order_id: ClassVar[int] = 8
    type: Literal["TaskInput"] = "TaskInput"
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
