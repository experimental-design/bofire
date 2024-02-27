from typing import List, Literal

import numpy as np
from pydantic import model_validator, validator

from bofire.data_models.features.api import DiscreteInput


class TaskInput(DiscreteInput):
    type: Literal["TaskInput"] = "TaskInput"
    n_tasks: int
    fidelities: List[int]

    @validator("fidelities")
    def validate_fidelities(cls, fidelities: List[int], values):
        # if fidelities is None:
        #    return [0 for _ in range(self.n_tasks)]
        if len(fidelities) != values["n_tasks"]:
            raise ValueError(
                "Length of fidelity lists must be equal to the number of tasks"
            )
        if list(set(fidelities)) != list(range(np.max(fidelities) + 1)):
            raise ValueError(
                "Fidelities must be a list containing integers, starting from 0 and increasing by 1"
            )
        return fidelities

    @model_validator(mode="before")
    def validate_values(cls, values):
        if "n_tasks" in values:
            values["values"] = list(range(values["n_tasks"]))
        return values
