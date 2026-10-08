from bofire.data_models.enum import REGRESSION_METRIC_DIRECTIONS, RegressionMetricsEnum


def test_all_regression_metrics_have_a_direction():
    assert REGRESSION_METRIC_DIRECTIONS == {
        RegressionMetricsEnum.MAE: "MINIMIZE",
        RegressionMetricsEnum.MSD: "MINIMIZE",
        RegressionMetricsEnum.MAPE: "MINIMIZE",
        RegressionMetricsEnum.FISHER: "MINIMIZE",
        RegressionMetricsEnum.R2: "MAXIMIZE",
        RegressionMetricsEnum.PEARSON: "MAXIMIZE",
        RegressionMetricsEnum.SPEARMAN: "MAXIMIZE",
    }
    assert set(REGRESSION_METRIC_DIRECTIONS) == set(RegressionMetricsEnum)
