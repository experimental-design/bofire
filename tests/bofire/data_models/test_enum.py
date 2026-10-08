from bofire.data_models.enum import REGRESSION_METRIC_DIRECTIONS, RegressionMetricsEnum


def test_all_regression_metrics_have_a_direction():
    assert set(REGRESSION_METRIC_DIRECTIONS) == set(RegressionMetricsEnum)
