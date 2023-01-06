import numpy as np
import pytest

from bofire.domain.features import CategoricalInput, ContinuousInput
from bofire.models.diagnostics import CVResult, CVResults, metrics


def generate_cvresult(key, num_samples):
    feature = ContinuousInput(key=key, lower_bound=10, upper_bound=20)
    observed = feature.sample(num_samples)
    predicted = observed + np.random.normal(0, 1, size=num_samples)
    return CVResult(key=key, observed=observed, predicted=predicted)


def test_cvresult_not_numeric():
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    feature2 = CategoricalInput(key="a", categories=["a", "b"])
    with pytest.raises(ValueError):
        CVResult(
            key=feature.key,
            observed=feature2.sample(num_samples),
            predicted=feature.sample(num_samples),
        )
    with pytest.raises(ValueError):
        CVResult(
            key=feature.key,
            observed=feature.sample(num_samples),
            predicted=feature2.sample(num_samples),
        )
    with pytest.raises(ValueError):
        CVResult(
            key=feature.key,
            observed=feature.sample(num_samples),
            predicted=feature.sample(num_samples),
            uncertainty=feature2.sample(num_samples),
        )


def test_cvresult_shape_mismatch():
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    with pytest.raises(ValueError):
        CVResult(
            key=feature.key,
            observed=feature.sample(5),
            predicted=feature.sample(6),
        )
    with pytest.raises(ValueError):
        CVResult(
            key=feature.key,
            observed=feature.sample(5),
            predicted=feature.sample(5),
            uncertainty=feature.sample(6),
        )


def test_cvresult_get_metric():
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv = CVResult(key=feature.key, observed=observed, predicted=predicted)
    assert cv.num_samples == 10
    for metric in metrics.keys():
        cv.get_metric(metric)


def test_cvresult_get_metric_invalid():
    num_samples = 1
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv = CVResult(key=feature.key, observed=observed, predicted=predicted)
    for metric in metrics.keys():
        with pytest.raises(ValueError):
            cv.get_metric(metric)


def test_cvresults_invalid():
    # test for empty results
    with pytest.raises(ValueError):
        CVResults(results=[])
    # test for wrong keys
    num_samples = 10
    feature = ContinuousInput(key="a", lower_bound=10, upper_bound=20)
    observed = feature.sample(n=num_samples)
    predicted = observed + np.random.normal(loc=0, scale=1, size=num_samples)
    cv1 = CVResult(key=feature.key, observed=observed, predicted=predicted)
    cv2 = CVResult(key="b", observed=observed, predicted=predicted)
    with pytest.raises(ValueError):
        CVResults(results=[cv1, cv2])


@pytest.mark.parametrize(
    "cv_results",
    [
        CVResults(
            results=[generate_cvresult(key="a", num_samples=10) for _ in range(10)]
        ),
        CVResults(
            results=[generate_cvresult(key="a", num_samples=10) for _ in range(5)]
        ),
    ],
)
def test_cvresults_get_metrics(cv_results):
    assert cv_results.key == "a"
    for metric in metrics:
        m = cv_results.get_metric(metric)
        assert len(m) == len(cv_results.results)
    df = cv_results.get_metrics()
    assert df.shape == (len(cv_results.results), len(metrics))


@pytest.mark.parametrize(
    "cv_results",
    [
        CVResults(
            results=[generate_cvresult(key="a", num_samples=1) for _ in range(10)]
        ),
        CVResults(
            results=[generate_cvresult(key="a", num_samples=1) for _ in range(5)]
        ),
    ],
)
def test_cvresults_get_metric_loo(cv_results):
    for metric in metrics:
        m = cv_results.get_metric(metric)
        assert len(m) == 1
    df = cv_results.get_metrics()
    assert df.shape == (1, len(metrics))


@pytest.mark.parametrize(
    "cv_results, expected",
    [
        (
            CVResults(
                results=[generate_cvresult(key="a", num_samples=1) for _ in range(5)]
            ),
            True,
        ),
        (
            CVResults(
                results=[generate_cvresult(key="a", num_samples=5) for _ in range(5)]
            ),
            False,
        ),
        (
            CVResults(
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
