import warnings
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator, validator
from scipy.integrate import simpson
from scipy.stats import fisher_exact, kendalltau, norm, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.domain import is_numeric
from bofire.data_models.enum import (
    ClassificationMetricsEnum,
    RegressionMetricsEnum,
    UQRegressionMetricsEnum,
)


def _accuracy_score(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the standard accuracy score.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: Accuracy score.

    """
    return float(accuracy_score(observed, predicted))


def _f1_score(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the f1 accuracy score.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: Accuracy score.

    """
    return float(f1_score(observed, predicted, average="micro"))


def _mean_absolute_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the mean absolute error.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: mean absolute error

    """
    return mean_absolute_error(observed, predicted)


def _mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the mean squared error.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: mean squared error

    """
    return mean_squared_error(observed, predicted)


def _mean_absolute_percentage_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the mean percentage error.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: mean percentage error

    """
    return mean_absolute_percentage_error(observed, predicted)


def _r2_score(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the R2 score.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: R2 score.

    """
    return float(r2_score(observed, predicted))


def _pearson(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the Pearson correlation coefficient.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: Pearson correlation coefficient.

    """
    with np.errstate(invalid="ignore"):
        rho, _ = pearsonr(predicted, observed)
    return float(rho)


def _spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the Spearman correlation coefficient.

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
            Ignored in the calculation. Defaults to None.

    Returns:
        float: Spearman correlation coefficient.

    """
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(predicted, observed)
    return float(rho)


def _fisher_exact_test_p(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Test if the model is able to distuinguish the bottom half of the
    observations from the top half.

    For this purpose Fisher's exact test is used together with the observations
    and predictions. The p value is returned. A low p value indicates that
    the model has some ability to distuiguish high from low values. A high p
    value indicates that the model cannot identify the difference or that the
    observations are too noisy to be able to tell.

    This implementation is taken from Ax: https://github.com/facebook/Ax/blob/main/ax/modelbridge/cross_validation.py

    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard
            deviation. Ignored in the calculation. Defaults to None.

    Returns:
        float: p value of the test.

    """
    n_half = len(observed) // 2
    top_obs = observed.argsort(axis=0)[-n_half:]
    top_est = predicted.argsort(axis=0)[-n_half:]
    # Construct contingency table
    tp = len(set(top_est).intersection(top_obs))
    fp = n_half - tp
    fn = n_half - tp
    tn = (len(observed) - n_half) - (n_half - tp)
    table = np.array([[tp, fp], [fn, tn]])
    # Compute the test statistic
    _, p = fisher_exact(table, alternative="greater")
    return float(p)


def _spearman_UQ(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the Spearman correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation.

    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.

    Returns:
        float: Spearman correlation coefficient.

    """
    if standard_deviation is None:
        warnings.warn(
            "Uncertainty quantification without standard deviation is not possible",
            UserWarning,
        )
        return np.nan
    ae = np.abs(observed - predicted)
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(ae, standard_deviation)
    return float(rho)


def _pearson_UQ(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the Pearson correlation coefficient between the models absolute error
    and the uncertainty - linear correlation.

    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.

    Returns:
        float: Pearson correlation coefficient.

    """
    if standard_deviation is None:
        warnings.warn(
            "Uncertainty quantification without standard deviation is not possible",
            UserWarning,
        )
        return np.nan
    ae = np.abs(observed - predicted)
    with np.errstate(invalid="ignore"):
        rho, _ = pearsonr(ae, standard_deviation)
    return float(rho)


def _kendall_UQ(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
) -> float:
    """Calculates the Kendall correlation coefficient between the models absolute error
    and the uncertainty - linear correlation.

    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.

    Returns:
        float: Kendall correlation coefficient.

    """
    if standard_deviation is None:
        warnings.warn(
            "Uncertainty quantification without standard deviation is not possible",
            UserWarning,
        )
        return np.nan
    ae = np.abs(observed - predicted)
    with np.errstate(invalid="ignore"):
        rho, _ = kendalltau(ae, standard_deviation)
    return float(rho)


def _CVPPDiagram(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
    num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Metric introduced in arXiv:2010.01118 [cs.LG] based on cross-validatory
    predictive p-values.
    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
        num_bins (int): number of bins for getting quantiles.

    Returns:
        np.ndarray: quantiles.
        np.ndarray: calibration score for each quantile.

    """
    if standard_deviation is None:
        raise ValueError(
            "Calibration metric without standard deviation is not possible",
        )
    lhs = np.abs((predicted - observed) / standard_deviation)
    qs = np.linspace(0, 1, num_bins)
    Cqs = np.empty(qs.shape)
    for ix, q in enumerate(qs):
        rhs = norm.ppf(((1.0 + q) / 2.0), loc=0.0, scale=1.0)
        Cqs[ix] = np.sum((lhs < rhs).astype(int)) / observed.shape[0]

    return qs, Cqs


def _MaximumMiscalibration(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
    num_bins: int = 10,
) -> float:
    """Miscalibration metric with CVPP
    WARNING - this metric only diagnoses systematic over- or under-
    confidence, i.e. a model that is overconfident for ~half of the
    quantiles and under-confident for ~half will still have a MiscalibrationArea
    of ~0.
    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
        num_bins (int): number of bins for getting quantiles.

    Returns:
        float: maximum miscalibration

    """
    try:
        qs, Cqs = _CVPPDiagram(
            observed=observed,
            predicted=predicted,
            standard_deviation=standard_deviation,
            num_bins=num_bins,
        )
        res = np.max(np.abs(Cqs - qs))

        return float(res)
    except ValueError:
        warnings.warn(
            "Calibration metric without standard deviation is not possible",
            UserWarning,
        )
        return np.nan


def _MiscalibrationArea(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
    num_bins: int = 10,
) -> float:
    """Miscalibration area metric with CVPP
    WARNING - this metric only diagnoses systematic over- or under-
    confidence, i.e. a model that is overconfident for ~half of the
    quantiles and under-confident for ~half will still have a MiscalibrationArea
    of ~0.
    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
        num_bins (int): number of bins for getting quantiles.

    Returns:
        float: total miscalibration area

    """
    try:
        qs, Cqs = _CVPPDiagram(
            observed=observed,
            predicted=predicted,
            standard_deviation=standard_deviation,
            num_bins=num_bins,
        )
        res = simpson(Cqs - qs, x=qs)

        return float(res)
    except ValueError:
        warnings.warn(
            "Calibration metric without standard deviation is not possible",
            UserWarning,
        )
        return np.nan


def _AbsoluteMiscalibrationArea(
    observed: np.ndarray,
    predicted: np.ndarray,
    standard_deviation: Optional[np.ndarray] = None,
    num_bins: int = 10,
) -> float:
    """Absolute miscalibration area metric with CVPP
    This implementation is taken from : https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/uncertainty_metrics.py


    Args:
        observed (np.ndarray): Observed data.
        predicted (np.ndarray): Predicted data.
        standard_deviation (Optional[np.ndarray], optional): Predicted standard deviation.
        num_bins (int): number of bins for getting quantiles.

    Returns:
        float: absolute miscalibration area

    """
    try:
        qs, Cqs = _CVPPDiagram(
            observed=observed,
            predicted=predicted,
            standard_deviation=standard_deviation,
            num_bins=num_bins,
        )
        res = simpson(np.abs(Cqs - qs), x=qs)

        return float(res)
    except ValueError:
        warnings.warn(
            "Calibration metric without standard deviation is not possible",
            UserWarning,
        )
        return np.nan


metrics = {
    RegressionMetricsEnum.MAE: _mean_absolute_error,
    RegressionMetricsEnum.MSD: _mean_squared_error,
    RegressionMetricsEnum.R2: _r2_score,
    RegressionMetricsEnum.MAPE: _mean_absolute_percentage_error,
    RegressionMetricsEnum.PEARSON: _pearson,
    RegressionMetricsEnum.SPEARMAN: _spearman,
    RegressionMetricsEnum.FISHER: _fisher_exact_test_p,
}

classification_metrics = {
    ClassificationMetricsEnum.ACCURACY: _accuracy_score,
    ClassificationMetricsEnum.F1: _f1_score,
}

UQ_metrics = {
    UQRegressionMetricsEnum.PEARSON_UQ: _pearson_UQ,
    UQRegressionMetricsEnum.SPEARMAN_UQ: _spearman_UQ,
    UQRegressionMetricsEnum.KENDALL_UQ: _kendall_UQ,
    UQRegressionMetricsEnum.MAXIMUMCALIBRATION: _MaximumMiscalibration,
    UQRegressionMetricsEnum.MISCALIBRATIONAREA: _MiscalibrationArea,
    UQRegressionMetricsEnum.ABSOLUTEMISCALIBRATIONAREA: _AbsoluteMiscalibrationArea,
}

all_metrics = {**metrics, **UQ_metrics, **classification_metrics}


class CvResult(BaseModel):
    """Container representing the results of one CV fold.

    Attributes:
        key (str): Key of the validated output feature.
        observed (pd.Series): Series holding the observed values
        predicted (pd.Series): Series holding the predicted values
        standard_deviation (pd.Series, optional): Series holding the standard deviation associated with
            the prediction. Defaults to None.
        labcodes (pd.Series, optional): Series holding the labcodes associated with the prediction.
            Defaults to None.
        X (pd.DataFrame, optional): DataFrame holding the input features associated with the prediction.
            Defaults to None.
        model_config (Dict, optional): Model configuration. Defaults to {"arbitrary_types_allowed": True}.

    """

    key: str
    observed: pd.Series
    predicted: pd.Series
    standard_deviation: Optional[pd.Series] = None
    labcodes: Optional[pd.Series] = None
    X: Optional[pd.DataFrame] = None
    model_config: Optional[Dict] = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_shapes(self):
        if not len(self.predicted) == len(self.observed):
            raise ValueError(
                f"Predicted values has length {len(self.predicted)} whereas observed has length {len(self.observed)}",
            )
        if self.standard_deviation is not None:
            if not len(self.predicted) == len(self.standard_deviation):
                raise ValueError(
                    f"Predicted values has length {len(self.predicted)} whereas standard_deviation has length {len(self.standard_deviation)}",
                )
        if self.labcodes is not None:
            if not len(self.predicted) == len(self.labcodes):
                raise ValueError(
                    f"Predicted values has length {len(self.predicted)} whereas labcodes has length {len(self.labcodes)}",
                )
        if self.X is not None:
            if not len(self.predicted) == len(self.X):
                raise ValueError(
                    f"Predicted values has length {len(self.predicted)} whereas X has length {len(self.X)}",
                )
        return self

    @field_validator("observed", "predicted")
    @classmethod
    def validate_series(cls, v, info):
        if not is_numeric(v):
            raise ValueError(f"Not all values of {info.field_name} are numerical")
        return v

    @field_validator("standard_deviation")
    @classmethod
    def validate_standard_deviation(cls, v, info):
        if v is None:
            return v
        if not is_numeric(v):
            raise ValueError(f"Not all values of {info.field_name} are numerical")
        return v

    @property
    def n_samples(self) -> int:
        """Returns the number of samples in the fold.

        Returns:
            int: Number of samples in the split.

        """
        return len(self.observed)

    def get_metric(
        self,
        metric: Union[
            ClassificationMetricsEnum,
            RegressionMetricsEnum,
            UQRegressionMetricsEnum,
        ],
    ) -> float:
        """Calculates a metric for the fold.

        Args:
            metric (RegressionMetricsEnum): Metric to calculate.

        Returns:
            float: Metric value.

        """
        if self.n_samples == 1:
            warnings.warn(
                "Metric cannot be calculated for only one sample. Null value will be returned",
            )
            return np.nan
        return all_metrics[metric](
            self.observed.values,
            self.predicted.values,
            self.standard_deviation,
        )


class CvResults(BaseModel):
    """Container holding all cv folds of a cross-validation run.

    Attributes:
        results (Sequence[CvResult]: Sequence of `CvResult` objects.

    """

    results: Sequence[CvResult]

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("results")
    def validate_results(cls, v, values):
        if len(v) <= 1:
            raise ValueError("`results` sequence has to contain at least two elements.")
        key = v[0].key
        for i in v:
            if i.key != key:
                raise ValueError("`CvResult` objects do not match.")
        for field in ["standard_deviation", "labcodes", "X"]:
            has_field = getattr(v[0], field) is not None
            for i in v:
                has_i = getattr(i, field) is not None
                if has_field != has_i:
                    raise ValueError(
                        f"Either all or none `CvResult` objects contain {field}.",
                    )
        # check columns of X
        if v[0].X is not None:
            cols = sorted(v[0].X.columns)
            for i in v:
                if sorted(i.X.columns) != cols:
                    raise ValueError("Columns of X do not match.")
        return v

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i) -> CvResult:
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
        return (np.array([r.n_samples for r in self.results]) == 1).all()

    def _combine_folds(self) -> CvResult:
        """Combines the `CvResult` splits into one flat array for predicted, observed and standard_deviation.

        Returns:
            Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]: One pd.Series for CvResult property.

        """
        observed = pd.concat([cv.observed for cv in self.results], ignore_index=True)
        predicted = pd.concat([cv.predicted for cv in self.results], ignore_index=True)
        if self.results[0].standard_deviation is not None:
            sd = pd.concat(
                [cv.standard_deviation for cv in self.results],
                ignore_index=True,
            )
        else:
            sd = None
        if self.results[0].labcodes is not None:
            labcodes = pd.concat(
                [cv.labcodes for cv in self.results],
                ignore_index=True,
            )
        else:
            labcodes = None
        if self.results[0].X is not None:
            X = pd.concat([cv.X for cv in self.results], ignore_index=True)
        else:
            X = None
        return CvResult(
            key=self.results[0].key,
            observed=observed,
            predicted=predicted,
            standard_deviation=sd,
            labcodes=labcodes,
            X=X,
        )

    def get_metric(
        self,
        metric: Union[
            ClassificationMetricsEnum,
            RegressionMetricsEnum,
            UQRegressionMetricsEnum,
        ],
        combine_folds: bool = True,
    ) -> pd.Series:
        """Calculates a metric for every fold and returns them as pd.Series.

        Args:
            metric (RegressionMetricsEnum): Metrics to calculate.
            combine_folds (bool, optional): If True the data in the split is combined before
                the metric is calculated. In this case only a single number is returned. If False
                the metric is calculated per fold. Defaults to True.

        Returns:
            pd.Series: Object containing the metric value for every fold.

        """
        if self.is_loo or combine_folds:
            return pd.Series(
                self._combine_folds().get_metric(metric=metric),
                name=metric.name,
            )
        return pd.Series(
            [cv.get_metric(metric) for cv in self.results],
            name=metric.name,
        )

    def get_metrics(
        self,
        metrics: Sequence[
            Union[
                ClassificationMetricsEnum,
                RegressionMetricsEnum,
                UQRegressionMetricsEnum,
            ]
        ] = [
            RegressionMetricsEnum.MAE,
            RegressionMetricsEnum.MSD,
            RegressionMetricsEnum.R2,
            RegressionMetricsEnum.MAPE,
            RegressionMetricsEnum.PEARSON,
            RegressionMetricsEnum.SPEARMAN,
            RegressionMetricsEnum.FISHER,
        ],
        combine_folds: bool = True,
    ) -> pd.DataFrame:
        """Calculates all metrics provided as list for every fold and returns all as pd.DataFrame.

        Args:
            metrics (Sequence[RegressionMetricsEnum], optional): Metrics to calculate. Defaults to R2, MAE, MSD, R2, MAPE.
            combine_folds (bool, optional): If True the data in the split is combined before
                the metric is calculated. In this case only a single number per metric is returned. If False
                the metric is calculated per fold. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe containing the metric values for all folds.

        """
        return pd.concat([self.get_metric(m, combine_folds) for m in metrics], axis=1)


# the following methods transform a CvResults object to a CrossValidationValues object
# in which the metrics are stored and not computed on the fly, moreover the field types
# are more backend friendly. It should be used to store CvResults in a backend system
class CrossValidationValues(BaseModel):
    observed: List[float] = Field(description="actual output values")
    predicted: List[float] = Field(description="predicted output values")
    standardDeviation: Optional[List[float]] = Field(
        description="standard deviation of predicted values",
        default=None,
    )
    metrics: Optional[Dict[str, float]] = Field(
        description="metrics per cv fold. Key is the metric type",
        default=None,
    )


def CvResults2CrossValidationValues(
    cv: CvResults,
) -> Dict[str, List[CrossValidationValues]]:
    cvResults = {cv.key: []}
    metrics = cv.get_metrics(combine_folds=False)
    for i, fold in enumerate(cv.results):
        cvResults[cv.key].append(
            CrossValidationValues(
                observed=fold.observed.tolist(),
                predicted=fold.predicted.tolist(),
                standardDeviation=(
                    fold.standard_deviation.tolist()
                    if fold.standard_deviation is not None
                    else None
                ),
                metrics=metrics.loc[i].to_dict() if fold.n_samples > 1 else None,
            ),
        )
    return cvResults
