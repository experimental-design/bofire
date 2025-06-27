from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.domain.domain import is_numeric
from bofire.data_models.features.api import CategoricalOutput
from bofire.data_models.surrogates.api import Surrogate as DataModel
from bofire.surrogates.values import PredictedValue
from bofire.utils.naming_conventions import (
    get_column_names,
    postprocess_categorical_predictions,
)


class Surrogate(ABC):
    def __init__(
        self,
        data_model: DataModel,
    ):
        self.inputs = data_model.inputs
        self.outputs = data_model.outputs
        self.input_preprocessing_specs = data_model.input_preprocessing_specs
        if data_model.dump is not None:
            self.loads(data_model.dump)
        else:
            self.model = None

    @classmethod
    def from_spec(cls, data_model: DataModel) -> "Surrogate":
        return cls(data_model=data_model)

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.model is not None

    def _prepare_data_for_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare the data for prediction."""
        # validate
        X = self.inputs.validate_experiments(X, strict=False)
        # transform
        Xt = self.inputs.transform(X, self.input_preprocessing_specs)
        # make sure that all is float64
        for c in Xt.columns:
            Xt[c] = pd.to_numeric(Xt[c], errors="raise")
        return Xt

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        # check if model is fitted
        if not self.is_fitted:
            raise ValueError("Model is not fitted/available yet.")
        # prepare data
        Xt = self._prepare_data_for_predict(experiments)
        # predict
        preds, stds = self._predict(Xt)
        # set up column names
        pred_cols, sd_cols = get_column_names(self.outputs)
        # postprocess
        predictions = pd.DataFrame(
            data=np.hstack((preds, stds)),
            columns=pred_cols + sd_cols,  # type: ignore
        )
        # append predictions for categorical cases
        predictions = postprocess_categorical_predictions(
            predictions=predictions,
            outputs=self.outputs,
        )
        # validate
        self.validate_predictions(predictions=predictions)
        # return
        return predictions

    def validate_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        # Get the column names
        pred_cols, sd_cols = get_column_names(self.outputs)
        expected_cols = pred_cols + sd_cols
        check_columns = list(expected_cols)
        for featkey in self.outputs.get_keys(CategoricalOutput):
            expected_cols = expected_cols + [f"{featkey}_{t}" for t in ["pred", "sd"]]
        if sorted(predictions.columns) != sorted(expected_cols):
            raise ValueError(
                f"Predictions are ill-formatted. Expected: {expected_cols}, got: {list(predictions.columns)}.",
            )
        # check that values are numeric
        if not is_numeric(predictions[check_columns]):
            raise ValueError("Not all values in predictions are numeric.")
        return predictions

    def to_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> Dict[str, List[PredictedValue]]:
        outputs = {key: [] for key in self.outputs.get_keys()}
        for _, row in predictions.iterrows():
            for key in self.outputs.get_keys():
                outputs[key].append(
                    PredictedValue(
                        predictedValue=row[f"{key}_pred"],
                        standardDeviation=row[f"{key}_sd"],
                    ),
                )
        return outputs

    @abstractmethod
    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def dumps(self) -> str:
        """Dumps the actual model to a string as this is not directly json serializable."""
        if not self.is_fitted:
            raise ValueError("Model has to be fitted before dumping")
        self._prepare_for_dump()
        return self._dumps()

    @abstractmethod
    def _dumps(self) -> str:
        pass

    def _prepare_for_dump(self):
        """Prepares the model before the dump."""

    @abstractmethod
    def loads(self, data: str):
        """Loads the actual model from a string and writes it to the `model` attribute."""
