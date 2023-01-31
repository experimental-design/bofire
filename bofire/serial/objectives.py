from typing import Union

from bofire.domain.objectives import (
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    SigmoidObjective,
    TargetObjective,
)

AnyObjective = Union[
    # IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    DeltaObjective,
    SigmoidObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    ConstantObjective,
    # AbstractTargetObjective,
    CloseToTargetObjective,
    TargetObjective,
]
