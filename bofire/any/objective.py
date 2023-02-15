from typing import Union

from bofire.domain import objectives

AnyObjective = Union[
    objectives.MaximizeObjective,
    objectives.MinimizeObjective,
    objectives.DeltaObjective,
    objectives.SigmoidObjective,
    objectives.MaximizeSigmoidObjective,
    objectives.MinimizeSigmoidObjective,
    objectives.ConstantObjective,
    objectives.CloseToTargetObjective,
    objectives.TargetObjective,
]

AnyAbstractObjective = Union[
    objectives.IdentityObjective,
    objectives.MaximizeObjective,
    objectives.MinimizeObjective,
    objectives.DeltaObjective,
    objectives.SigmoidObjective,
    objectives.MaximizeSigmoidObjective,
    objectives.MinimizeSigmoidObjective,
    objectives.ConstantObjective,
    objectives.AbstractTargetObjective,
    objectives.CloseToTargetObjective,
    objectives.TargetObjective,
    objectives.BotorchConstrainedObjective,
]
