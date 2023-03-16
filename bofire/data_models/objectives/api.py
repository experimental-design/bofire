from typing import Union

from bofire.data_models.objectives.contant import ConstantObjective
from bofire.data_models.objectives.identity import (
    DeltaObjective,
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.objectives.objective import Objective
from bofire.data_models.objectives.sigmoid import (
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    SigmoidObjective,
)
from bofire.data_models.objectives.target import (
    BotorchConstrainedObjective,
    CloseToTargetObjective,
    TargetObjective,
)

AbstractObjective = Union[
    Objective,
    IdentityObjective,
    SigmoidObjective,
    BotorchConstrainedObjective,
]

AnyObjective = Union[
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MinimizeObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
    CloseToTargetObjective,
]
