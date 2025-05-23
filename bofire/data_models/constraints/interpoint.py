import math
from typing import Annotated, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.constraints.constraint import Constraint
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput


class InterpointConstraint(Constraint):
    """An interpoint constraint describes required relationships between individual
    candidates when asking a strategy for returning more than one candidate.
    """

    type: str


class InterpointEqualityConstraint(InterpointConstraint):
    """Constraint that forces that values of a certain feature of a set/batch of
    candidates should have the same value. Currently this is only implemented for ContinuousInput features.

    Attributes:
        feature(str): The constrained feature.
        multiplicity(int): The multiplicity of the constraint, stating how many
            values of the feature in the batch should have always the same value.

    """

    type: Literal["InterpointEqualityConstraint"] = "InterpointEqualityConstraint"
    features: Annotated[List[str], Field(min_length=1), Field(max_length=1)]
    multiplicity: Optional[Annotated[int, Field(ge=2)]] = None

    @property
    def feature(self) -> str:
        """Feature to be constrained."""
        return self.features[0]

    def validate_inputs(self, inputs: Inputs):
        if self.feature not in inputs.get_keys(ContinuousInput):
            raise ValueError(
                f"Feature {self.feature} is not a continuous input feature in the provided Inputs object.",
            )

    def is_fulfilled(
        self,
        experiments: pd.DataFrame,
        tol: Optional[float] = 1e-6,
    ) -> pd.Series:
        multiplicity = self.multiplicity or len(experiments)
        for i in range(math.ceil(len(experiments) / multiplicity)):
            batch = experiments[self.feature].values[
                i * multiplicity : min((i + 1) * multiplicity, len(experiments))
            ]
            if not np.allclose(batch, batch[0]):
                return pd.Series([False])
        return pd.Series([True])

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        """Numerically evaluates the constraint. Returns the distance to the constraint fulfillment
        for each batch of size batch_size.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint on.

        Returns:
            pd.Series: Distance to reach constraint fulfillment.

        """
        multiplicity = self.multiplicity or len(experiments)
        n_batches = int(np.ceil(experiments.shape[0] / multiplicity))
        feature_values = np.zeros(n_batches * multiplicity)
        feature_values[: experiments.shape[0]] = experiments[self.feature].values
        feature_values[experiments.shape[0] :] = feature_values[-multiplicity]
        feature_values = feature_values.reshape(n_batches, multiplicity).T

        batchwise_constraint_matrix = np.zeros(shape=(multiplicity - 1, multiplicity))
        batchwise_constraint_matrix[:, 0] = 1.0
        batchwise_constraint_matrix[:, 1:] = -np.eye(multiplicity - 1)

        return pd.Series(
            np.linalg.norm(batchwise_constraint_matrix @ feature_values, axis=0, ord=2)
            ** 2,
            index=[f"batch_{i}" for i in range(n_batches)],
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Method `jacobian` currently not implemented.")

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[str, int], pd.DataFrame]:
        raise NotImplementedError("Method `hessian` currently not implemented.")
