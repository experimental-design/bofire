import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from bofire.domain.features import CategoricalInput, ContinuousInput
from bofire.models.diagnostics import (
    CvResult,
    CvResults,
    _mean_absolute_error,
    _mean_absolute_percentage_error,
    _mean_squared_error,
    _pearson,
    _r2_score,
    _spearman,
    metrics,
)
from bofire.utils.enum import RegressionMetricsEnum


def generate_cvresult(key, num_samples):
    feature = ContinuousInput(key=key, lower_bound=10, upper_bound=20)
    observed = feature.sample(num_samples)
    predicted = observed + np.random.normal(0, 1, size=num_samples)
    return CvResult(key=key, observed=observed, predicted=predicted)


@pytest.mark.parametrize(
    "bofire, sklearn",
    [
        (_mean_absolute_error, mean_absolute_error),
        (_mean_absolute_percentage_error, mean_absolute_percentage_error),
        (_mean_squared_error, mean_squared_error),
        (_r2_score, r2_score),
    ],
)
def test_sklearn_metrics(bofire, sklearn):
    num_samples = 20
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(num_samples).values
    predicted = observed + np.random.normal(0, 1, size=num_samples)
    sd = np.random.normal(0, 1, size=num_samples)
    assert bofire(observed, predicted, sd) == sklearn(observed, predicted)
    assert bofire(observed, predicted) == sklearn(observed, predicted)


@pytest.mark.parametrize(
    "bofire, scipy",
    [
        (_pearson, pearsonr),
        (_spearman, spearmanr),
    ],
)
def test_scipy_metrics(bofire, scipy):
    num_samples = 20
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(num_samples).values
    predicted = observed + np.random.normal(0, 1, size=num_samples)
    sd = np.random.normal(0, 1, size=num_samples)
    s, _ = scipy(predicted, observed)
    assert bofire(observed, predicted, sd) == s
    assert bofire(observed, predicted) == s


def test_cvresult_not_numeric():
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    feature2 = CategoricalInput(key="a", categories=["a", "b"])
    with pytest.raises(ValueError):
        CvResult(
            key=feature.key,
            observed=feature2.sample(num_samples),
            predicted=feature.sample(num_samples),
        )
    with pytest.raises(ValueError):
        CvResult(
            key=feature.key,
            observed=feature.sample(num_samples),
            predicted=feature2.sample(num_samples),
        )
    with pytest.raises(ValueError):
        CvResult(
            key=feature.key,
            observed=feature.sample(num_samples),
            predicted=feature.sample(num_samples),
            standard_deviation=feature2.sample(num_samples),
        )


def test_cvresult_shape_mismatch():
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    with pytest.raises(ValueError):
        CvResult(
            key=feature.key,
            observed=feature.sample(5),
            predicted=feature.sample(6),
        )
    with pytest.raises(ValueError):
        CvResult(
            key=feature.key,
            observed=feature.sample(5),
            predicted=feature.sample(5),
            standard_deviation=feature.sample(6),
        )


def test_cvresult_get_metric():
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv = CvResult(key=feature.key, observed=observed, predicted=predicted)
    assert cv.num_samples == 10
    for metric in metrics.keys():
        cv.get_metric(metric)


def test_cvresult_get_metric_invalid():
    num_samples = 1
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv = CvResult(key=feature.key, observed=observed, predicted=predicted)
    for metric in metrics.keys():
        with pytest.raises(ValueError):
            cv.get_metric(metric)


def test_cvresults_invalid():
    # test for empty results
    with pytest.raises(ValueError):
        CvResults(results=[])
    # test for wrong keys
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv1 = CvResult(key=feature.key, observed=observed, predicted=predicted)
    cv2 = CvResult(key="b", observed=observed, predicted=predicted)
    with pytest.raises(ValueError):
        CvResults(results=[cv1, cv2])
    cv1 = CvResult(key=feature.key, observed=observed, predicted=predicted)
    cv2 = CvResult(
        key="b", observed=observed, predicted=predicted, standard_deviation=observed
    )
    with pytest.raises(ValueError):
        CvResults(results=[cv1, cv2])


@pytest.mark.parametrize(
    "cv_results",
    [
        CvResults(
            results=[generate_cvresult(key="a", num_samples=10) for _ in range(10)]
        ),
        CvResults(
            results=[generate_cvresult(key="a", num_samples=10) for _ in range(5)]
        ),
    ],
)
def test_cvresults_get_metrics(cv_results):
    assert cv_results.key == "a"
    for metric in metrics:
        for combine_folds in [True, False]:
            m = cv_results.get_metric(metric, combine_folds)
            if combine_folds:
                assert len(m) == 1
            else:
                assert len(m) == len(cv_results.results)
    for combine_folds in [True, False]:
        df = cv_results.get_metrics(combine_folds=combine_folds)
        if combine_folds:
            assert df.shape == (1, len(metrics))
        else:
            assert df.shape == (len(cv_results.results), len(metrics))


def test_cvresults_get_metric_combine_folds():
    cv_results = CvResults(
        results=[generate_cvresult(key="a", num_samples=10) for _ in range(10)]
    )
    assert np.allclose(
        cv_results.get_metric(RegressionMetricsEnum.MAE, combine_folds=True).values[0],
        cv_results.get_metric(RegressionMetricsEnum.MAE, combine_folds=False).mean(),
    )


def test_cvresults_combine_folds():
    cv_results = CvResults(
        results=[
            generate_cvresult(key="a", num_samples=5),
            generate_cvresult(key="a", num_samples=6),
        ]
    )
    observed, predicted, _ = cv_results._combine_folds()
    assert observed.shape == (11,)
    assert predicted.shape == (11,)


@pytest.mark.parametrize(
    "cv_results",
    [
        CvResults(
            results=[generate_cvresult(key="a", num_samples=1) for _ in range(10)]
        ),
        CvResults(
            results=[generate_cvresult(key="a", num_samples=1) for _ in range(5)]
        ),
    ],
)
def test_cvresults_get_metrics_loo(cv_results):
    observed, predicted, _ = cv_results._combine_folds()
    assert observed.shape == (len(cv_results),)
    assert predicted.shape == (len(cv_results),)
    for metric in metrics:
        m = cv_results.get_metric(metric)
        assert len(m) == 1
    df = cv_results.get_metrics()
    assert df.shape == (1, len(metrics))


@pytest.mark.parametrize(
    "cv_results, expected",
    [
        (
            CvResults(
                results=[generate_cvresult(key="a", num_samples=1) for _ in range(5)]
            ),
            True,
        ),
        (
            CvResults(
                results=[generate_cvresult(key="a", num_samples=5) for _ in range(5)]
            ),
            False,
        ),
        (
            CvResults(
                results=[
                    generate_cvresult(key="a", num_samples=5),
                    generate_cvresult(key="a", num_samples=1),
                ]
            ),
            False,
        ),
    ],
)
def test_cvresults_is_loo(cv_results, expected):
    assert cv_results.is_loo == expected
