from typing import Union

from bofire.data_models.objectives.categorical import ConstrainedCategoricalObjective
from bofire.data_models.objectives.desirabilities import (
    DecreasingDesirabilityObjective,
    DesirabilityObjective,
    IncreasingDesirabilityObjective,
    InRangeDesirability,
    PeakDesirabilityObjective,
)
from bofire.data_models.objectives.identity import (
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.objectives.objective import Objective
from bofire.data_models.objectives.sigmoid import (
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    MovingMaximizeSigmoidObjective,
    SigmoidObjective,
)
from bofire.data_models.objectives.target import (
    CloseToTargetObjective,
    ConstrainedObjective,
    TargetObjective,
)


AbstractObjective = Union[
    Objective,
    IdentityObjective,
    SigmoidObjective,
    ConstrainedObjective,
    DesirabilityObjective,
]

AnyCategoricalObjective = ConstrainedCategoricalObjective

AnyConstraintObjective = Union[
    MaximizeSigmoidObjective,
    MovingMaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
]

AnyRealObjective = Union[
    MaximizeObjective,
    MinimizeObjective,
    CloseToTargetObjective,
    IncreasingDesirabilityObjective,
    DecreasingDesirabilityObjective,
    PeakDesirabilityObjective,
    InRangeDesirability,
]

AnyObjective = Union[
    MaximizeObjective,
    MinimizeObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
    CloseToTargetObjective,
    ConstrainedCategoricalObjective,
    MovingMaximizeSigmoidObjective,
    IncreasingDesirabilityObjective,
    DecreasingDesirabilityObjective,
    PeakDesirabilityObjective,
    InRangeDesirability,
]
