from typing import Union

from bofire.domain.objectives import (
    AbstractTargetObjective,
    BotorchConstrainedObjective,
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    IdentityObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    SigmoidObjective,
    TargetObjective,
)

AnyObjective = Union[
    MaximizeObjective,
    MinimizeObjective,
    DeltaObjective,
    SigmoidObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    ConstantObjective,
    CloseToTargetObjective,
    TargetObjective,
]

AnyAbstractObjective = Union[
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    DeltaObjective,
    SigmoidObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    ConstantObjective,
    AbstractTargetObjective,
    CloseToTargetObjective,
    TargetObjective,
    BotorchConstrainedObjective,
]
