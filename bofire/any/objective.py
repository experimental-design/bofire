from typing import Union

from bofire.domain import objective

AnyObjective = Union[
    objective.MaximizeObjective,
    objective.MinimizeObjective,
    objective.DeltaObjective,
    objective.SigmoidObjective,
    objective.MaximizeSigmoidObjective,
    objective.MinimizeSigmoidObjective,
    objective.ConstantObjective,
    objective.CloseToTargetObjective,
    objective.TargetObjective,
]

AnyAbstractObjective = Union[
    objective.IdentityObjective,
    objective.MaximizeObjective,
    objective.MinimizeObjective,
    objective.DeltaObjective,
    objective.SigmoidObjective,
    objective.MaximizeSigmoidObjective,
    objective.MinimizeSigmoidObjective,
    objective.ConstantObjective,
    objective.AbstractTargetObjective,
    objective.CloseToTargetObjective,
    objective.TargetObjective,
    objective.BotorchConstrainedObjective,
]
