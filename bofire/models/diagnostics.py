from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import root_validator, validator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from bofire.domain.util import PydanticBaseModel, is_numeric
from bofire.utils.enum import RegressionMetricsEnum

metrics = {
    RegressionMetricsEnum.MAE: mean_absolute_error,
    RegressionMetricsEnum.MSD: mean_squared_error,
    RegressionMetricsEnum.R2: r2_score,
    RegressionMetricsEnum.MAPE: mean_absolute_percentage_error,
}


class CVResult(PydanticBaseModel):
    """Container representing the results of one CV fold.

    Attributes:
        key (str): Key of the validated output feature.
        observed (pd.Series): Series holding the observed values
        predicted (pd.Series): Series holding the predicted values
        uncertainty (pd.Series, optional): Series holding the uncertainty associated with
            the prediction. Defaults to None.
    """

    key: str
    observed: pd.Series
    predicted: pd.Series
    uncertainty: Optional[pd.Series] = None

    @root_validator(pre=True)
    def validate_shapes(cls, values):
        if not len(values["predicted"]) == len(values["observed"]):
            raise ValueError(
                f"Predicted values has length {len(values['predicted'])} whereas observed has length {len(values['observed'])}"
            )
        if "uncertainty" in values:
            if not len(values["predicted"]) == len(values["uncertainty"]):
                raise ValueError(
                    f"Predicted values has length {len(values['predicted'])} whereas uncertainty has length {len(values['uncertainty'])}"
                )
        return values

    @validator("observed")
    def validate_observed(cls, v, values):
        if not is_numeric(v):
            raise ValueError("Not all values of observed are numerical")
        return v

    @validator("predicted")
    def validate_predicted(cls, v, values):
        if not is_numeric(v):
            raise ValueError("Not all values of observed are numerical")
        return v

    @validator("uncertainty")
    def validate_uncertainty(cls, v, values):
        if not is_numeric(v):
            raise ValueError("Not all values of observed are numerical")
        return v

    @property
    def num_samples(self) -> int:
        """Returns the number of samples in the fold.

        Returns:
            int: Number of samples in the split.
        """
        return len(self.observed)

    def get_metric(self, metric: RegressionMetricsEnum) -> float:
        """Calculates a metric for the fold.

        Args:
            metric (RegressionMetricsEnum): Metric to calculate.

        Returns:
            float: Metric value.
        """
        if self.num_samples == 1:
            raise ValueError("Metric cannot be calculated for only one sample.")
        return metrics[metric](self.observed.values, self.predicted.values)


class CVResults(PydanticBaseModel):
    """Container holding all cv folds of a cross-validation run.

    Attributes:
        results (Sequence[CVResult]: Sequence of `CVResult` objects.
    """

    results: Sequence[CVResult]

    @validator("results")
    def validate_results(cls, v, values):
        if len(v) <= 1:
            raise ValueError("`results` sequence has to contain at least two elements.")
        key = v[0].key
        for i in v:
            if i.key != key:
                raise ValueError("`CVResult` objects do not match.")
        return v

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i) -> CVResult:
        return self.results[i]

    @property
    def key(self) -> str:
        """Returns name of the feature for which the cross validation was performed.

        Returns:
            str: feature name.
        """
        return self.results[0].key

    @property
    def is_loo(self) -> bool:
        """Checks if the object represents a LOO-CV

        Returns:
            bool: True if LOO-CV else False.
        """
        return (np.array([r.num_samples for r in self.results]) == 1).all()

    def get_metric(self, metric: RegressionMetricsEnum) -> pd.Series:
        """Calculates a metric for every fold and returns them as pd.Series.

        Args:
            metric (RegressionMetricsEnum): Metrics to calculate.

        Returns:
            pd.Series: Object containing the metric value for every fold.
        """
        if self.is_loo:
            return pd.Series(
                [
                    metrics[metric](
                        [cv.observed.values[0] for cv in self.results],
                        [cv.predicted.values[0] for cv in self.results],
                    )
                ],
                name=metric.name,
            )
        return pd.Series(
            [cv.get_metric(metric) for cv in self.results], name=metric.name
        )

    def get_metrics(
        self,
        metrics: Sequence[RegressionMetricsEnum] = [
            RegressionMetricsEnum.MAE,
            RegressionMetricsEnum.MSD,
            RegressionMetricsEnum.R2,
            RegressionMetricsEnum.MAPE,
        ],
    ) -> pd.DataFrame:
        """Calculates all metrics provided as list for every fold and returns all as pd.DataFrame.

        Args:
            metrics (Sequence[RegressionMetricsEnum], optional): Metrics to calculate. Defaults to R2, MAE, MSD, R2, MAPE.

        Returns:
            pd.DataFrame: Dataframe containing the metric values for all folds.
        """
        return pd.concat([self.get_metric(m) for m in metrics], axis=1)
