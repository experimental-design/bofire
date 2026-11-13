from abc import abstractmethod
from typing import Annotated, Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.base import BaseModel


TGt0 = Annotated[float, Field(gt=0)]
TGe0 = Annotated[float, Field(ge=0)]


class Objective(BaseModel):
    """The base class for all objectives"""

    type: Any

    @model_validator(mode="before")
    @classmethod
    def validate_no_weight(cls, data: Any) -> Any:
        """Drops the former weight `w` of stored objectives if it is the neutral 1;
        any other weight has to be set on the scalarization instead."""
        if isinstance(data, dict) and "w" in data:
            data = dict(data)
            if data.pop("w") != 1:
                raise ValueError(
                    "Objective weights moved to the scalarization of SoboStrategy, "
                    "e.g. WeightedSumScalarization(weights={...}).",
                )
        return data

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
