from typing import List, Tuple

import pandas as pd

from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import CategoricalOutput, ContinuousOutput


def get_column_names(outputs: Outputs) -> Tuple[List[str], List[str]]:
    """Specifies column names for given Outputs type.

    Args:
        outputs (Outputs): The Outputs object containing the individual outputs.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the prediction column names and the standard deviation column names

    """
    pred_cols, sd_cols = [], []
    for featkey in outputs.get_keys(CategoricalOutput):
        pred_cols = pred_cols + [
            f"{featkey}_{cat}_prob"
            for cat in outputs.get_by_key(featkey).categories  # type: ignore
        ]
        sd_cols = sd_cols + [
            f"{featkey}_{cat}_sd"
            for cat in outputs.get_by_key(featkey).categories  # type: ignore
        ]
    for featkey in outputs.get_keys(ContinuousOutput):
        pred_cols = pred_cols + [f"{featkey}_pred"]
        sd_cols = sd_cols + [f"{featkey}_sd"]

    return pred_cols, sd_cols


def postprocess_categorical_predictions(
    predictions: pd.DataFrame,
    outputs: Outputs,
) -> pd.DataFrame:
    """Postprocess categorical predictions by finding the maximum probability location

    Args:
        predictions (pd.DataFrame): The dataframe containing the predictions.
        outputs (Outputs): The Outputs object containing the individual outputs.

    Returns:
        predictions (pd.DataFrame): The (potentially modified) original dataframe with categorical predictions added

    """
    for feat in outputs.get():
        if isinstance(feat, CategoricalOutput):
            predictions.insert(
                loc=0,
                column=f"{feat.key}_pred",
                value=predictions.filter(regex=f"{feat.key}(.*)_prob")
                .idxmax(1)
                .str.replace(f"{feat.key}_", "")
                .str.replace("_prob", "")
                .values,  # type: ignore
            )
            predictions.insert(
                loc=1,
                column=f"{feat.key}_sd",
                value=0.0,
            )
    return predictions
