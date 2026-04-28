import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold

from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective
from bofire.surrogates.diagnostics import CvResult, CvResults
from bofire.surrogates.surrogate import Surrogate


class TrainableSurrogate(ABC):
    """Mixin for surrogates that can be trained. Concrete subclasses must also
    inherit from :class:`Surrogate`, which provides ``inputs``, ``outputs``,
    and ``predict``."""

    # These attributes are provided by Surrogate via multiple inheritance.
    inputs: Inputs
    outputs: Outputs
    predict: Callable[..., pd.DataFrame]

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def fit(self, experiments: pd.DataFrame, options: Optional[Dict] = None):
        """
        Fit the surrogate model to the provided experiments.

        Args:
            experiments (pd.DataFrame): The experimental data to fit the model.
            options (Optional[Dict], optional): Additional options for fitting the model. Defaults to None.
        """
        # validate
        experiments = self.inputs.validate_experiments(experiments, strict=False)
        experiments = self.outputs.validate_experiments(experiments)
        # preprocess
        experiments = self._preprocess_experiments(experiments)
        X = experiments[self.inputs.get_keys()]
        # TODO: output feature validation
        Y = experiments[self.outputs.get_keys()]
        # fit
        options = options or {}
        self._fit(X=X, Y=Y, **options)

    def _preprocess_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the experiments based on the output filtering setting.

        Args:
            experiments (pd.DataFrame): The experimental data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed experimental data.
        """
        if self._output_filtering is None:
            return experiments
        if self._output_filtering == OutputFilteringEnum.ALL:
            return self.outputs.preprocess_experiments_all_valid_outputs(
                experiments=experiments,
                output_feature_keys=self.outputs.get_keys(),
            )
        if self._output_filtering == OutputFilteringEnum.ANY:
            return self.outputs.preprocess_experiments_any_valid_outputs(  # ty: ignore[unresolved-attribute]
                experiments=experiments,
                output_feature_keys=self.outputs.get_keys(),
            )
        raise ValueError("Unknown output filtering option requested.")

    @abstractmethod
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        """
        Abstract method to fit the model to the provided data.

        Args:
            X (pd.DataFrame): The input features.
            Y (pd.DataFrame): The output targets.
        """
        pass

    def cross_validate(
        self,
        experiments: pd.DataFrame,
        folds: int = -1,
        include_X: bool = False,
        include_labcodes: bool = False,
        random_state: Optional[int] = None,
        stratified_feature: Optional[str] = None,
        group_split_column: Optional[str] = None,
        hooks: Optional[
            Dict[
                str,
                Callable[
                    [
                        Surrogate,
                        pd.DataFrame,
                        pd.DataFrame,
                        pd.DataFrame,
                        pd.DataFrame,
                    ],
                    Any,
                ],
            ]
        ] = None,
        hook_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[CvResults, CvResults, Dict[str, List[Any]]]:
        """Perform a cross validation for the provided training data.

        Args:
            experiments (pd.DataFrame): Data on which the cross validation should be performed.
            folds (int, optional): Number of folds. -1 is equal to LOO CV. Defaults to -1.
            include_X (bool, optional): If true the X values of the fold are written to respective CvResult objects for
                later analysis. Defaults to False.
            random_state (int, optional): Controls the randomness of the indices in the train and test sets of each fold.
                Defaults to None.
            stratified_feature (str, optional): The feature name to preserve the percentage of samples for each class in
                the stratified folds. Defaults to None.
            group_split_column (str, optional): The column name of the group id.
                This parameter is used to ensure that the splits are made such that the same group is not present in both
                training and testing sets. This is useful in scenarios where data points are related or dependent on each
                other, and splitting them into different sets would violate the assumption of independence. The number of
                unique groups must be greater than or equal to the number of folds. Defaults to None.
            hooks (Dict[str, Callable[[Model, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], Any]], optional):
                Dictionary of callable hooks that are called within the CV loop. The callable retrieves the current trained
                modeld and the current CV folds in the following order: X_train, y_train, X_test, y_test. Defaults to {}.
            hook_kwargs (Dict[str, Dict[str, Any]], optional): Dictionary holding hook specific keyword arguments.
                Defaults to {}.

        Returns:
            Tuple[CvResults, CvResults, Dict[str, List[Any]]]: First CvResults object reflects the training data,
                second CvResults object the test data, dictionary object holds the return values of the applied hooks.

        """
        if include_labcodes and "labcode" not in experiments.columns:
            raise ValueError("No labcodes available for the provided experiments.")

        if len(self.outputs) > 1:
            raise NotImplementedError(
                "Cross validation not implemented for multi-output models",
            )

        if stratified_feature is not None:
            if stratified_feature not in (
                self.inputs.get_keys() + self.outputs.get_keys()
            ):
                raise ValueError(
                    "The feature to be stratified is not in the model inputs or outputs",
                )
            try:
                feat = self.inputs.get_by_key(stratified_feature)
            except KeyError:
                feat = self.outputs.get_by_key(stratified_feature)
            if not isinstance(
                feat,
                (DiscreteInput, CategoricalInput, CategoricalOutput, ContinuousOutput),
            ):
                raise ValueError(
                    "The feature to be stratified needs to be a DiscreteInput, CategoricalInput, CategoricalOutput, or ContinuousOutput",
                )

        if group_split_column is not None:
            # check if the group split column is present in the experiments
            if group_split_column not in experiments.columns:
                raise ValueError(
                    f"Group split column {group_split_column} is not present in the experiments."
                )
            ngroups = len(experiments[group_split_column].unique())
            # check if the number of unique groups is greater than or equal to the number of folds
            if ngroups < folds:
                raise ValueError(
                    f"Number of unique groups {ngroups} is less than the number of folds {folds}."
                )

        # first filter the experiments based on the model setting
        experiments = self._preprocess_experiments(experiments)
        n = len(experiments)
        folds = self._check_valid_nfolds(folds, n)
        # preprocess hooks
        if hooks is None:
            hooks = {}
        if hook_kwargs is None:
            hook_kwargs = {}
        hook_results = {key: [] for key in hooks.keys()}

        # instantiate kfold object
        cv, cv_func = self._make_cv_split(
            experiments,
            folds,
            stratified_feature=stratified_feature,
            group_split_column=group_split_column,
            random_state=random_state,
        )

        key = self.outputs.get_keys()[0]
        train_results = []
        test_results = []
        # now get the indices for the split
        for train_index, test_index in cv_func:
            X_train = experiments.iloc[train_index][self.inputs.get_keys()]
            X_test = experiments.iloc[test_index][self.inputs.get_keys()]
            y_train = experiments.iloc[train_index][self.outputs.get_keys()]
            y_test = experiments.iloc[test_index][self.outputs.get_keys()]
            train_labcodes = (
                experiments.iloc[train_index]["labcode"] if include_labcodes else None
            )
            test_labcodes = (
                experiments.iloc[test_index]["labcode"] if include_labcodes else None
            )
            # now fit the model
            self._fit(X_train, y_train)
            # now do the scoring
            y_test_pred = self.predict(X_test)
            y_train_pred = self.predict(X_train)

            # Convert to categorical if applicable
            objective = self.outputs.get_by_key(key)
            if isinstance(objective, ConstrainedCategoricalObjective):
                y_test_pred[f"{key}_pred"] = y_test_pred[f"{key}_pred"].map(
                    objective.to_dict_label(),
                )
                y_train_pred[f"{key}_pred"] = y_train_pred[f"{key}_pred"].map(
                    objective.to_dict_label(),
                )
                y_test[key] = y_test[key].map(
                    objective.to_dict_label(),
                )
                y_train[key] = y_train[key].map(
                    objective.to_dict_label(),
                )

            # now store the results
            train_results.append(
                CvResult(
                    key=key,
                    observed=y_train[key],
                    predicted=y_train_pred[key + "_pred"],
                    standard_deviation=y_train_pred[key + "_sd"],
                    X=X_train if include_X else None,
                    labcodes=train_labcodes,
                ),
            )
            test_results.append(
                CvResult(
                    key=key,
                    observed=y_test[key],
                    predicted=y_test_pred[key + "_pred"],
                    standard_deviation=y_test_pred[key + "_sd"],
                    X=X_test if include_X else None,
                    labcodes=test_labcodes,
                ),
            )
            # now call the hooks if available
            for hookname, hook in hooks.items():
                hook_results[hookname].append(
                    hook(  # ty: ignore[missing-argument]
                        surrogate=self,  # ty: ignore[unknown-argument]
                        X_train=X_train,  # ty: ignore[unknown-argument]
                        y_train=y_train,  # ty: ignore[unknown-argument]
                        X_test=X_test,  # ty: ignore[unknown-argument]
                        y_test=y_test,  # ty: ignore[unknown-argument]
                        **hook_kwargs.get(hookname, {}),
                    ),
                )
        return (
            CvResults(results=train_results),
            CvResults(results=test_results),
            hook_results,
        )

    def _check_valid_nfolds(self, folds, n):
        """
        Check and adjust the number of folds for cross-validation.

        Args:
            folds (int): The requested number of folds.
            n (int): The number of experiments.

        Returns:
            int: The adjusted number of folds.

        Raises:
            ValueError: If the number of folds is invalid or if the experiments are empty.
        """
        if n == 0:
            raise ValueError("Experiments is empty.")
        if folds > n:
            warnings.warn(
                f"Training data only has {n} experiments, which is less than folds, fallback to LOOCV.",
            )
            folds = n
        elif folds < 2 and folds != -1:
            raise ValueError("Folds must be -1 for LOO, or > 1.")
        elif folds == -1:
            folds = n
        return folds

    def _make_cv_split(
        self,
        experiments: pd.DataFrame,
        folds: int,
        stratified_feature: Optional[str] = None,
        group_split_column: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[
        Union[KFold, StratifiedKFold, GroupShuffleSplit],
        Generator[Tuple[np.ndarray, np.ndarray], None, None],
    ]:
        """
        Create the cross-validation split object.

        Args:
            experiments (pd.DataFrame): The experimental data.
            folds (int): The number of folds.
            random_state (Optional[int]): The random state for reproducibility.
            stratified_feature (Optional[str]): The feature to stratify by.
            group_split_column (Optional[str]): The column for group splitting.

        Returns:
            Tuple: The cross-validation split object and the split function.
        """
        if stratified_feature is None:
            if group_split_column is not None:
                # GROUP SPLIT FUNCTIONALITY
                cv = GroupShuffleSplit(n_splits=folds, random_state=random_state)
                cv_func = cv.split(experiments, groups=experiments[group_split_column])
            else:
                cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
                cv_func = cv.split(experiments)
        else:
            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            )
            cv_func = cv.split(
                experiments.drop([stratified_feature], axis=1),
                experiments[stratified_feature],
            )
        return cv, cv_func
