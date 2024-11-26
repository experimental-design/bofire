import pandas as pd
import pytest

from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import CategoricalOutput, ContinuousOutput
from bofire.data_models.objectives.api import (
    ConstrainedCategoricalObjective,
    MinimizeObjective,
)
from bofire.utils.naming_conventions import (
    get_column_names,
    postprocess_categorical_predictions,
)


continuous_output = ContinuousOutput(key="cont", objective=MinimizeObjective(w=1))
categorical_output = CategoricalOutput(
    key="cat",
    categories=["alpha", "beta"],
    objective=ConstrainedCategoricalObjective(
        categories=["alpha", "beta"],
        desirability=[True, False],
    ),
)
predictions = pd.DataFrame(
    data=[[0.8, 0.2, 1.5, 1e-3, 1e-2, 1e-1]],
    columns=[
        "cat_alpha_prob",
        "cat_beta_prob",
        "cont_pred",
        "cat_alpha_sd",
        "cat_beta_sd",
        "cont_sd",
    ],
)


@pytest.mark.parametrize(
    "output_features, expected_names",
    [
        ([continuous_output], ["cont_pred", "cont_sd"]),
        (
            [categorical_output],
            ["cat_alpha_prob", "cat_beta_prob", "cat_alpha_sd", "cat_beta_sd"],
        ),
        (
            [continuous_output, categorical_output],
            [
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
        ),
        (
            [categorical_output, continuous_output],
            [
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
        ),
    ],
)
def test_get_column_names(output_features, expected_names):
    test_outputs = Outputs(features=output_features)
    pred_cols, sd_cols = get_column_names(test_outputs)
    assert pred_cols + sd_cols == expected_names


@pytest.mark.parametrize(
    "output_features, input_names, final_names",
    [
        ([continuous_output], ["cont_pred", "cont_sd"], ["cont_pred", "cont_sd"]),
        (
            [categorical_output],
            ["cat_alpha_prob", "cat_beta_prob", "cat_alpha_sd", "cat_beta_sd"],
            [
                "cat_pred",
                "cat_sd",
                "cat_alpha_prob",
                "cat_beta_prob",
                "cat_alpha_sd",
                "cat_beta_sd",
            ],
        ),
        (
            [continuous_output, categorical_output],
            [
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
            [
                "cat_pred",
                "cat_sd",
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
        ),
        (
            [categorical_output, continuous_output],
            [
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
            [
                "cat_pred",
                "cat_sd",
                "cat_alpha_prob",
                "cat_beta_prob",
                "cont_pred",
                "cat_alpha_sd",
                "cat_beta_sd",
                "cont_sd",
            ],
        ),
    ],
)
def test_postprocess_categorical_predictions(output_features, input_names, final_names):
    test_outputs = Outputs(features=output_features)
    updated_preds = postprocess_categorical_predictions(
        predictions=predictions[input_names],
        outputs=test_outputs,
    )
    assert updated_preds.columns.tolist() == final_names
