from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from pydantic import PositiveInt

from bofire.data_models.features.task import TaskInput
from bofire.data_models.strategies.api import Strategy as DataModel
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.data_models.candidate import Candidate
from bofire.strategies.data_models.values import InputValue, OutputValue
from bofire.strategies.strategy import Strategy
from bofire.surrogates.feature_importance import shap_importance
from bofire.utils.naming_conventions import (
    get_column_names,
    postprocess_categorical_predictions,
)


class PredictiveStrategy(Strategy):
    """Base class for all model based strategies.

    Provides abstract scaffold for fit, predict, and calc_acquistion methods.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if the strategy is fitted."""
        return self._is_fitted

    @property
    @abstractmethod
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        pass

    def _postprocess_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the candidates.

        Here we append the predictions to the candidates.

        Args:
            candidates: The generated candidates.

        Returns:
            Dataframe with candidates and corresponding predictions
        """
        preds = self.predict(candidates)
        return pd.concat((candidates, preds), axis=1)

    def ask(
        self,
        candidate_count: Optional[PositiveInt] = None,
        add_pending: bool = False,
        raise_validation_error: bool = True,
    ) -> pd.DataFrame:
        """Function to generate new candidates.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to
                be generated. If not provided, the number of candidates is
                determined automatically. Defaults to None.
            add_pending (bool, optional): If true the proposed candidates are
                added to the set of pending experiments. Defaults to False.
            raise_validation_error (bool, optional): If true an error will be
                raised if candidates violate constraints, otherwise only a
                warning will be displayed. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)

        """
        candidates = super().ask(
            candidate_count=candidate_count,
            add_pending=add_pending,
            raise_validation_error=raise_validation_error,
        )
        self.domain.validate_candidates(
            candidates=candidates,
            raise_validation_error=raise_validation_error,
        )
        return candidates

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
        retrain: bool = True,
    ):
        """This function passes new experimental data to the optimizer.

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data
                should replace the former dataFrame or if the new experiments
                should be attached. Defaults to False.
            retrain (bool, optional): If True, model(s) are retrained when new
                experimental data is passed to the optimizer. Defaults to True.

        """
        # maybe unite the preprocessor here with the one of the parent tell
        # TODO: add self.domain.validate_experiments(self.experiments, strict=True) here to ensure variance in each feature?
        if len(experiments) == 0:
            return
        if replace:
            self.set_experiments(experiments)
        else:
            self.add_experiments(experiments)
        # we check here that the experiments do not have completely fixed columns
        cleaned_experiments = (
            self.domain.outputs.preprocess_experiments_all_valid_outputs(
                experiments=experiments,
            )
        )

        fixed_nontasks = (
            feat
            for feat in self.domain.inputs.get_fixed()
            if not isinstance(feat, TaskInput)
        )
        for feature in fixed_nontasks:
            fixed_value = feature.fixed_value()
            assert fixed_value is not None
            if (cleaned_experiments[feature.key] == fixed_value[0]).all():
                raise ValueError(
                    f"No variance in experiments for fixed feature {feature.key}",
                )
        if retrain and self.has_sufficient_experiments():
            self.fit()
            # we have a separate _tell here for things that are relevant when
            # setting up the strategy but unrelated to fitting the models like
            # initializing the ACQF.
            self._tell()

    def calc_shap(self, candidates: pd.DataFrame) -> Dict[str, shap.Explanation]:
        """Calculate SHAP values for the provided candidates.

        Args:
            candidates (pd.DataFrame): Candidates for which SHAP values should
                be calculated.

        Returns:
            Dict[str, shap.Explanation]: Dictionary with SHAP values for each
                output feature.

        """
        if not self.is_fitted:
            raise ValueError("Model not yet fitted.")
        if self.experiments is None:
            raise ValueError(
                "No experiments available. Strategy needs experiments to perform "
                "SHAP calculations. Use `tell` to provide experimental data.",
            )

        return shap_importance(
            predictor=self, bg_experiments=self.experiments, experiments=candidates
        )

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for the provided experiments. Only input features
        have to be provided.

        Args:
            experiments (pd.DataFrame): Experimental data for which predictions
                should be performed.

        Returns:
            pd.DataFrame: Dataframe with the predicted values.

        """
        if self.is_fitted is not True:
            raise ValueError("Model not yet fitted.")
        if self.experiments is None:
            raise ValueError(
                "No experiments available. Strategy needs experiments to perform "
                "predictions. Use `tell` to provide experimental data.",
            )
        # TODO: validate also here the experiments but only for the input_columns
        # transformed = self.transformer.transform(experiments)
        transformed = self.domain.inputs.transform(
            experiments=experiments,
            specs=self.input_preprocessing_specs,
        )
        preds, stds = self._predict(transformed)
        pred_cols, sd_cols = get_column_names(self.domain.outputs)
        if stds is not None:
            predictions = pd.DataFrame(
                data=np.hstack((preds, stds)),
                columns=pred_cols + sd_cols,
            )
        else:
            predictions = pd.DataFrame(
                data=preds,
                columns=pred_cols,
            )
        predictions = postprocess_categorical_predictions(
            predictions=predictions,
            outputs=self.domain.outputs,
        )
        objectives = self.domain.outputs(
            predictions, experiments_adapt=self.experiments, predictions=True
        )
        predictions = pd.concat((predictions, objectives), axis=1)
        predictions.index = experiments.index
        return predictions

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Abstract method in which the actual prediction is happening. Has to
        be overwritten.
        """

    def fit(self):
        """Fit the model(s) to the experimental data."""
        assert (
            self.experiments is not None and len(self.experiments) > 0
        ), "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        # transformed = self.transformer.fit_transform(self.experiments)
        self._fit(self.experiments)
        self._is_fitted = True

    @abstractmethod
    def _fit(self, experiments: pd.DataFrame):
        """Abstract method where the actual prediction are occurring."""

    def to_candidates(self, candidates: pd.DataFrame) -> List[Candidate]:
        """Transform candiadtes dataframe to a list of `Candidate` objects.

        Args:
            candidates (pd.DataFrame): candidates formatted as dataframe

        Returns:
            List[Candidate]: candidates formatted as list of `Candidate` objects.

        """
        return [
            Candidate(
                inputValues={
                    key: InputValue(value=str(row[key]))
                    for key in self.domain.inputs.get_keys()
                },
                outputValues={
                    feat.key: OutputValue(
                        predictedValue=str(row[f"{feat.key}_pred"]),
                        standardDeviation=row[f"{feat.key}_sd"],
                        objective=(
                            row[f"{feat.key}_des"]
                            if feat.objective is not None
                            else 1.0
                        ),
                    )
                    for feat in self.domain.outputs.get()
                },
            )
            for _, row in candidates.iterrows()
        ]
