from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import model_validator

from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.feature import Input


class TaskInput(Input):
    """Abstract base class for task-encoding inputs.

    This class is not directly instantiable and is not part of any
    ``AnyFeature``/``AnyInput`` union. Use :class:`CategoricalTaskInput` or
    :class:`ContinuousTaskInput` instead. It exists solely so that strategies
    can use ``isinstance(feat, TaskInput)`` to detect either flavour.

    Task inputs carry no descriptor data — see :meth:`validate_no_descriptor_data`.
    """

    type: Any

    @model_validator(mode="after")
    def validate_no_descriptor_data(self):
        """Reject ``descriptors`` / ``structure`` on a task input.

        A task input is an *index*: it says which task or fidelity an observation came
        from. The relationship between tasks is something the surrogate **learns** (the
        inter-task covariance of a ``MultiTaskGP``), not something read off descriptor
        columns, and no kernel in BoFire consumes task descriptors. Structures are
        meaningless outright — a task is not a molecule.

        The fields are inherited from ``CategoricalInput`` / ``ContinuousInput``, which
        do describe real entities; without this guard a surrogate could silently
        descriptor-encode a task index. Should task descriptors ever be wanted, they
        should be introduced deliberately, together with a kernel that uses them.
        """
        # both fields are inherited, so read them defensively; "empty" is `{}` for
        # descriptors and `None` for structure, and falsy covers either.
        if getattr(self, "descriptors", None):
            raise ValueError(
                f"{self.key}: task inputs cannot carry `descriptors`. A task input is "
                "an index into a set of tasks; inter-task correlation is learned by "
                "the surrogate, not derived from descriptor columns.",
            )
        if getattr(self, "structure", None):
            raise ValueError(
                f"{self.key}: task inputs cannot carry a `structure` column; a task "
                "is not a molecule.",
            )
        return self


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
