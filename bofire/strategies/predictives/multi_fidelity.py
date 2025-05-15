from typing import List, cast

import numpy as np
import pandas as pd
from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
)
from bofire.data_models.api import Domain
from bofire.data_models.features.api import TaskInput
from bofire.data_models.outlier_detection.outlier_detections import OutlierDetections
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy as DataModel,
)
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.strategies.strategy import make_strategy
from bofire.utils.naming_conventions import get_column_names


class MultiFidelityStrategy(SoboStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.task_feature_key = self.domain.inputs.get_keys(TaskInput)[0]

        ft = data_model.fidelity_thresholds
        M = len(self.domain.inputs.get_by_key(self.task_feature_key).fidelities)  # type: ignore
        self.fidelity_thresholds = ft if isinstance(ft, list) else [ft] * M

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        """Generate new candidates (x, m).

        This is a greedy optimization of the acquisition function. We first
        optimize the acqf for the target fidelity to generate a candidate x,
        then select the lowest fidelity that has a variance exceeding a
        threshold.

        Args:
            candidate_count (int): number of candidates to be generated

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if candidate_count > 1:
            raise NotImplementedError("Batch optimization is not yet implemented")

        self._verify_all_fidelities_observed()

        task_feature: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        # only optimize the input x on the target fidelity
        # we fix the fidelity by setting all other fidelities to 'not allowed'
        prev_allowed = task_feature.allowed
        task_feature.allowed = [fidelity == 0 for fidelity in task_feature.fidelities]
        x = super()._ask(candidate_count)
        task_feature.allowed = prev_allowed
        fidelity_pred = self._select_fidelity_and_get_predict(x)
        x.update(fidelity_pred)
        return x

    def _select_fidelity_and_get_predict(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """Select the fidelity for a given input.

        Uses the variance based approach (see [Kandasamy et al. 2016,
        Folch et al. 2023]) to select the lowest fidelity that has a variance
        exceeding a threshold. If no such fidelity exists, pick the target fidelity

        Args:
            X (pd.DataFrame): optimum input of target fidelity

        Returns:
            pd.DataFrame: selected fidelity and prediction
        """
        fidelity_input: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        assert self.model is not None and self.experiments is not None
        assert fidelity_input.allowed is not None

        sorted_fidelities = np.argsort(fidelity_input.fidelities)[::-1]
        target_fidelity_idx = sorted_fidelities[-1]
        target_fidelity = fidelity_input.fidelities[target_fidelity_idx]
        _, sd_cols = get_column_names(self.domain.outputs)

        for fidelity_idx in sorted_fidelities:
            if not fidelity_input.allowed[fidelity_idx]:
                continue

            m = fidelity_input.fidelities[fidelity_idx]
            fidelity_name = fidelity_input.categories[fidelity_idx]

            fidelity_threshold_scale = self.model.outcome_transform.stdvs.item()
            fidelity_threshold = self.fidelity_thresholds[m] * fidelity_threshold_scale

            X_fid = X.assign(**{self.task_feature_key: fidelity_name})
            transformed = self.domain.inputs.transform(
                experiments=X_fid, specs=self.input_preprocessing_specs
            )
            pred = self.predict(transformed)

            if (pred[sd_cols] > fidelity_threshold).all().all() or m == target_fidelity:
                pred[self.task_feature_key] = fidelity_name
                return pred

    def _verify_all_fidelities_observed(self) -> None:
        """Get all fidelities that have at least one observation.

        We use this instead of overriding `has_sufficient_experiments` to provide
        a more descriptive error message."""
        assert self.experiments is not None
        observed_fidelities = set(self.experiments[self.task_feature_key].unique())
        allowed_fidelities = set(
            self.domain.inputs.get_by_key(
                self.task_feature_key
            ).get_allowed_categories()  # type: ignore
        )
        missing_fidelities = allowed_fidelities - observed_fidelities
        if missing_fidelities:
            raise ValueError(f"Some tasks have no experiments: {missing_fidelities}")

    @classmethod
    def make(  # type: ignore
        cls,
        domain: Domain,
        fidelity_thresholds: List[float] | float | None = None,
        acquisition_function: AnySingleObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """
        Create a new instance of the multi-fidelity optimization strategy with the given parameters. This strategy
        is useful if you have different measurement fidelities that measure the same thing with different cost and accuracy.
        As an example, you can have a simulation that is fast but inaccurate and the real experiment that is slow and expensive,
        but more accurate.

        K. Kandasamy, G. Dasarathy, J. B. Oliva, J. Schneider, B. Póczos.
        Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations.
        Advances in Neural Information Processing Systems, 29, 2016.

        Jose Pablo Folch, Robert M Lee, Behrang Shafei, David Walz, Calvin Tsay, Mark van der Wilk, Ruth Misener.
        Combining Multi-Fidelity Modelling and Asynchronous Batch Bayesian Optimization.
        Computers & Chemical Engineering Volume 172, 2023.

        Args:
            domain: The optimization domain of the strategy.
            fidelity_thresholds: The thresholds for the fidelity. If a single value is provided, it will be used for all fidelities.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The acquisition optimizer to use.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of outlier checks.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation.
            seed: The random seed to use.
        """
        return cast(Self, make_strategy(cls, DataModel, locals()))
