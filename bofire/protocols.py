from typing import Protocol

import pandas as pd

from bofire.data_models.domain.api import Inputs, Outputs


class Predictor(Protocol):
    """Protocol for a predictor, which can be either a surrogate or a predictive strategy."""

    @property
    def is_fitted(self) -> bool:
        """Check if the predictor is fitted."""
        ...

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Predict using the predictor."""
        ...

    @property
    def inputs(self) -> Inputs:
        """Get the inputs of the predictor."""
        ...

    @property
    def outputs(self) -> Outputs:
        """Get the outputs of the predictor."""
        ...
