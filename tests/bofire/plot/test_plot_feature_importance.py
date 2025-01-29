import pandas as pd
import plotly.graph_objs as go
import pytest

from bofire.plot.feature_importance import plot_feature_importance_by_feature_plotly


sensitivity_values = {
    "MAE": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": -21.92325805224506, "std": 5.703349112330737},
            "x_2": {"mean": -37.05931404256329, "std": 5.613756792793183},
            "x_3": {"mean": 2.0572606288737916e-05, "std": 4.700467949580387e-06},
        },
    ),
    "MSD": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": -1913.8378145981662, "std": 611.3022576277862},
            "x_2": {"mean": -3284.137666801092, "std": 654.4911332798574},
            "x_3": {"mean": 0.0004564972101661624, "std": 0.00011682042637203013},
        },
    ),
    "R2": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": 0.7080295685024562, "std": 0.22615295318723475},
            "x_2": {"mean": 1.2149757714010154, "std": 0.24213079663812434},
            "x_3": {"mean": -1.6888239984247377e-07, "std": 4.321803833441057e-08},
        },
    ),
    "MAPE": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": -0.25222530028551865, "std": 0.03489127935370138},
            "x_2": {"mean": -0.5370619811730817, "std": 0.11369772721431769},
            "x_3": {"mean": 5.487732457898353e-07, "std": 8.822306360202737e-08},
        },
    ),
    "PEARSON": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": 0.3994021861753644, "std": 0.13086633591708324},
            "x_2": {"mean": 0.7592436114073695, "std": 0.21430456729157352},
            "x_3": {"mean": -7.317327355149672e-08, "std": 3.843313979276787e-08},
        },
    ),
    "SPEARMAN": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": 0.5260606060606061, "std": 0.10443955989594189},
            "x_2": {"mean": 0.5115151515151515, "std": 0.19247935264270746},
            "x_3": {"mean": 1.1102230246251565e-16, "std": 1.1102230246251565e-16},
        },
    ),
    "FISHER": pd.DataFrame.from_dict(
        {
            "x_1": {"mean": -0.3373015873015873, "std": 0.1944039478399348},
            "x_2": {"mean": -0.49603174603174616, "std": 0.0},
            "x_3": {"mean": 0.0, "std": 0.0},
        },
    ),
}


@pytest.mark.parametrize(
    "relative, show_std, caption, importance_measure",
    [
        (relative, show_std, caption, importance_measure)
        for relative in [True, False]
        for show_std in [True, False]
        for caption in ["", "caption"]
        for importance_measure in ["", "permutation"]
    ],
)
def test_feature_importances(relative, show_std, caption, importance_measure):
    plot = plot_feature_importance_by_feature_plotly(
        sensitivity_values=sensitivity_values,
        relative=relative,
        show_std=show_std,
        caption=caption,
        importance_measure=importance_measure,
    )
    assert isinstance(plot, go.Figure)
