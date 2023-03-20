from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.base import BaseModel


class AcquisitionFunction(BaseModel):
    """
    A class representing an acquisition function used in Bayesian optimization.

    Parameters:
    -----------
    type : str
        The type of the acquisition function to be used, such as "qEI" (Expected Improvement) or "qUCB" (Upper Confidence Bound).
    """

    type: str


class qNEI(AcquisitionFunction):
    """Acquisition function of type "qNEI" (Noisy Expected Improvement)"""

    type: Literal["qNEI"] = "qNEI"


class qEI(AcquisitionFunction):
    """Acquisition function of type "qEI" (Expected Improvement)"""

    type: Literal["qEI"] = "qEI"


class qSR(AcquisitionFunction):
    """Acquisition function of type "qSR" (Simple Regret)"""

    type: Literal["qSR"] = "qSR"


class qUCB(AcquisitionFunction):

    """Acquisition function of type "qUCB" (Upper Confidence Bound)"""

    type: Literal["qUCB"] = "qUCB"
    beta: PositiveFloat = 0.2


class qPI(AcquisitionFunction):
    """Acquisition function of type "qPI" (Probability of Improvement)"""

    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = 1e-3
