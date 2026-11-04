from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.feature import Input


class TaskInput(Input):
    """Abstract base class for task-encoding inputs.

    This class is not directly instantiable and is not part of any
    ``AnyFeature``/``AnyInput`` union. Use :class:`CategoricalTaskInput` or
    :class:`ContinuousTaskInput` instead. It exists solely so that strategies
    can use ``isinstance(feat, TaskInput)`` to detect either flavour.

    Task inputs carry no descriptor data: both flavours narrow ``descriptors`` to
    ``None``, so it cannot be set.
    """

    type: Any


class CategoricalTaskInput(TaskInput, CategoricalInput):
    """A categorical index over tasks. Carries no descriptor data.

    A task input says *which* task an observation came from; the relationship between
    tasks is what the surrogate learns (the inter-task covariance of a ``MultiTaskGP``),
    not something read off descriptor columns. ``descriptors`` is therefore narrowed to
    ``None`` — the constraint is in the type, and visible in the schema.

    Examples:
        An accurate task and a cheaper approximation of it. Note that the target task
        is the one at fidelity 0:

        >>> CategoricalTaskInput(
        ...     key="task",
        ...     categories=["experiment", "simulation"],
        ...     fidelities=[0, 1],
        ... )
    """

    order_id: ClassVar[int] = 8
    type: Literal["CategoricalTaskInput"] = "CategoricalTaskInput"
    descriptors: None = Field(
        default=None,
        description="Always None. A task input carries no descriptor data: the "
        "relationship between tasks is learned as the inter-task covariance of the "
        "surrogate, not read off descriptor columns.",
    )
    fidelities: list[int] = Field(
        default=[],
        description="Fidelity level of each task, one entry per category in the same "
        "order. Level 0 is the target task, the accurate and expensive one being "
        "optimized for; higher levels are progressively cheaper, less accurate "
        "approximations. The levels used must run from 0 upwards without gaps, though "
        "several tasks may share a level. Defaults to every task being at level 0.",
    )

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
    """A continuous fidelity parameter. Carries no descriptor data (see
    :class:`CategoricalTaskInput`).

    Examples:
        >>> ContinuousTaskInput(key="mesh_resolution", bounds=(0.0, 1.0))
    """

    order_id: ClassVar[int] = 11
    type: Literal["ContinuousTaskInput"] = "ContinuousTaskInput"  # type: ignore
    descriptors: None = Field(
        default=None,
        description="Always None. A task input carries no descriptor data: the "
        "relationship between fidelities is learned by the surrogate, not read off "
        "descriptor columns.",
    )
