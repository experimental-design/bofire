from abc import abstractmethod
from typing import Annotated, Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel


TGt0 = Annotated[float, Field(gt=0)]
TGe0 = Annotated[float, Field(ge=0)]
TWeight = Annotated[float, Field(gt=0, le=1)]


class Objective(BaseModel):
    """The base class for all objectives"""

    type: Any
    w: TWeight = Field(
        default=1.0,
        description="Relative weight of this objective. Only the strategies that "
        "aggregate several objectives into one acquisition function read it, namely "
        "the additive, multiplicative and custom SOBO strategies; the others ignore "
        "it.",
    )

    @abstractmethod
    def to_description(self) -> str:
        """Return a human-readable description of this objective."""

    @abstractmethod
    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """Abstract method to define the call function for the class Objective

        Args:
            x (np.ndarray): An array of x values for which the objective should be evaluated.
            x_adapt (Optional[np.ndarray], optional): An array of x values which are used to
                update the objective parameters on the fly. Defaults to None.

        Returns:
            np.ndarray: The desirability of the passed x values

        """


# TODO: should this inherit from Objective?
class ConstrainedObjective:
    """This abstract class offers a convenience routine for transforming sigmoid based objectives to botorch output constraints."""
