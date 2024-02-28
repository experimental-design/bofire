from typing import Annotated, List, Literal, Optional

import numpy as np
from pydantic import Field, field_validator

from bofire.data_models.features.categorical import CategoricalInput


class TaskInput(CategoricalInput):
    type: Literal["TaskInput"] = "TaskInput"
    fidelities: Optional[Annotated[List[int], Field(validate_default=True)]] = None

    @field_validator("fidelities")
    def validate_fidelities(cls, fidelities: List[int], info):
        n_tasks = len(info.data["categories"])
        if fidelities is None:
            return [0 for _ in range(n_tasks)]
        if len(fidelities) != n_tasks:
            raise ValueError(
                "Length of fidelity lists must be equal to the number of tasks"
            )
        if list(set(fidelities)) != list(range(np.max(fidelities) + 1)):
            raise ValueError(
                "Fidelities must be a list containing integers, starting from 0 and increasing by 1"
            )
        return fidelities
