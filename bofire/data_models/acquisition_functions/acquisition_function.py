from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionFunction(BaseModel):
    type: Any


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    type: Any


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    type: Any


class qNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qNEI"] = "qNEI"
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qLogNEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogNEI"] = "qLogNEI"
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qEI"] = "qEI"
    n_mc_samples: IntPowerOfTwo = 512


class qLogEI(SingleObjectiveAcquisitionFunction):
    type: Literal["qLogEI"] = "qLogEI"
    n_mc_samples: IntPowerOfTwo = 512


class qSR(SingleObjectiveAcquisitionFunction):
    type: Literal["qSR"] = "qSR"
    n_mc_samples: IntPowerOfTwo = 512


class qUCB(SingleObjectiveAcquisitionFunction):
    type: Literal["qUCB"] = "qUCB"
    beta: Annotated[float, Field(ge=0)] = 0.2
    n_mc_samples: IntPowerOfTwo = 512


class qPI(SingleObjectiveAcquisitionFunction):
    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3
    n_mc_samples: IntPowerOfTwo = 512


class qEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qEHVI"] = "qEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    n_mc_samples: IntPowerOfTwo = 512


class qLogEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogEHVI"] = "qLogEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    n_mc_samples: IntPowerOfTwo = 512


class qNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qNEHVI"] = "qNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qLogNEHVI(MultiObjectiveAcquisitionFunction):
    type: Literal["qLogNEHVI"] = "qLogNEHVI"
    alpha: Annotated[float, Field(ge=0)] = 0.0
    prune_baseline: bool = True
    n_mc_samples: IntPowerOfTwo = 512


class qNegIntPosVar(SingleObjectiveAcquisitionFunction):
    type: Literal["qNegIntPosVar"] = "qNegIntPosVar"
    n_mc_samples: IntPowerOfTwo = 512
    weights: Optional[Dict[str, PositiveFloat]] = Field(default_factory=lambda: None)


class qLogPF(SingleObjectiveAcquisitionFunction):
    """MC based batch LogProbability of Feasibility acquisition function.

    It is used to select the next batch of experiments to maximize the
    probability of finding feasible solutions with respect to output
    constraints in the next batch. It can be only used in the SoboStrategy
    and is especially useful in combination with the FeasibleExperimentCondition
    within the StepwiseStrategy.

    Attributes:
        n_mc_samples: Number of Monte Carlo samples to use to
            approximate the probability of feasibility.

    """

    type: Literal["qLogPF"] = "qLogPF"
    n_mc_samples: IntPowerOfTwo = 512
