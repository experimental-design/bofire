from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from pydantic import Field, validator
from sklearn.model_selection import KFold

from bofire.domain.features import InputFeatures, OutputFeatures, TInputTransformSpecs
from bofire.domain.util import PydanticBaseModel
from bofire.models.diagnostics import CVResult, CVResults
from bofire.utils.enum import OutputFilteringEnum


class Model(PydanticBaseModel):

    input_features: InputFeatures
    output_features: OutputFeatures
    input_preprocessing_specs: TInputTransformSpecs = Field(default_factory=lambda: {})

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        # we also validate the number of input features here
        if len(values["input_features"]) == 0:
            raise ValueError("At least one input feature has to be provided.")
        v = values["input_features"]._validate_transform_specs(v)
        return v

    @validator("output_features")
    def validate_output_features(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one output feature has to be provided.")
        return v

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # validate
        X = self.input_features.validate_experiments(X, strict=False)
        # transform
        Xt = self.input_features.transform(X, self.input_preprocessing_specs)
        # predict
        preds, stds = self._predict(Xt)
        # postprocess
        return pd.DataFrame(
            data=np.hstack((preds, stds)),
            columns=["%s_pred" % featkey for featkey in self.output_features.get_keys()]
            + ["%s_sd" % featkey for featkey in self.output_features.get_keys()],
        )

    @abstractmethod
    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass


class TrainableModel:

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def fit(self, experiments: pd.DataFrame):
        # preprocess
        experiments = self._preprocess_experiments(experiments)
        # validate
        experiments = self.input_features.validate_experiments(  # type: ignore
            experiments, strict=False
        )
        X = experiments[self.input_features.get_keys()]  # type: ignore
        # TODO: output feature validation
        Y = experiments[self.output_features.get_keys()]  # type: ignore
        # fit
        self._fit(X=X, Y=Y)  # type: ignore

    def _preprocess_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if self._output_filtering is None:
            return experiments
        elif self._output_filtering == OutputFilteringEnum.ALL:
            return self.output_features.preprocess_experiments_all_valid_outputs(  # type: ignore
                experiments=experiments,
                output_feature_keys=self.output_features.get_keys(),  # type: ignore
            )
        elif self._output_filtering == OutputFilteringEnum.ANY:
            return self.output_features.preprocess_experiments_any_valid_outputs(  # type: ignore
                experiments=experiments,
                output_feature_keys=self.output_features.get_keys(),  # type: ignore
            )
        else:
            raise ValueError("Unknown output filtering option requested.")

    @abstractmethod
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def cross_validate(
        self, experiments: pd.DataFrame, folds: int = -1
    ) -> Tuple[CVResults, CVResults]:
        """Perform a cross validation for the provided training data.

        Args:
            experiments (pd.DataFrame): Data on which the cross validation should be performed.
            folds (int, optional): Number of folds. -1 is equal to LOO CV. Defaults to -1.

        Returns:
            Tuple[CVResults, CVResults]: First CVResults object reflects the training data,
                second CVResults object the test data
        """
        if len(self.output_features) > 1:  # type: ignore
            raise NotImplementedError(
                "Cross validation not implemented for multi-output models"
            )
        n = len(experiments)
        if folds > n:
            raise ValueError(
                f"Training data only has {n} experiments, which is less than folds"
            )
        elif n == 0:
            raise ValueError("Experiments is empty.")
        elif folds < 2 and folds != -1:
            raise ValueError("Folds must be -1 for LOO, or > 1.")
        elif folds == -1:
            folds = n
        # instantiate kfold object
        cv = KFold(n_splits=folds, shuffle=True)
        key = self.output_features.get_keys()[0]  # type: ignore
        # first filter the experiments based on the model setting
        experiments = self._preprocess_experiments(experiments)
        train_results = []
        test_results = []
        # now get the indices for the split
        for train_index, test_index in cv.split(experiments):
            X_train = experiments.loc[train_index, self.input_features.get_keys()]  # type: ignore
            X_test = experiments.loc[test_index, self.input_features.get_keys()]  # type: ignore
            y_train = experiments.loc[train_index, self.output_features.get_keys()]  # type: ignore
            y_test = experiments.loc[test_index, self.output_features.get_keys()]  # type: ignore
            # now fit the model
            self._fit(X_train, y_train)
            # now do the scoring
            y_test_pred = self.predict(X_test)  # type: ignore
            y_train_pred = self.predict(X_train)  # type: ignore
            # now store the results
            train_results.append(
                CVResult(  # type: ignore
                    key=key,
                    observed=y_train[key],
                    predicted=y_train_pred[key + "_pred"],
                    standard_deviation=y_train_pred[key + "_sd"],
                )
            )
            test_results.append(
                CVResult(  # type: ignore
                    key=key,
                    observed=y_test[key],
                    predicted=y_test_pred[key + "_pred"],
                    standard_deviation=y_test_pred[key + "_sd"],
                )
            )
        return CVResults(results=train_results), CVResults(results=test_results)
