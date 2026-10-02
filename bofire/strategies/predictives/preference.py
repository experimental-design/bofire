from typing import Literal, Optional

import pandas as pd
import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.sampling.normal import SobolQMCNormalSampler
from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import (
    AnyPreferenceAcquisitionFunction,
    qEUBO,
)
from bofire.data_models.api import Domain
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import PreferenceStrategy as DataModel
from bofire.data_models.strategies.convergence_criteria.api import (
    AnyConvergenceCriterion,
)
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.surrogates.api import BotorchSurrogates as SurrogateDataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.strategy import make_strategy
from bofire.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.surrogates.pairwise_gp import PairwiseGPSurrogate


class PreferenceStrategy(BotorchStrategy):
    """Preferential Bayesian optimization using a pairwise GP."""

    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self._preferences: Optional[pd.DataFrame] = None

        self.surrogates = BotorchSurrogates(data_model=self.surrogate_specs)
        surrogate = self.surrogates.surrogates[0]
        if not isinstance(surrogate, PairwiseGPSurrogate):
            raise TypeError("PreferenceStrategy requires a PairwiseGPSurrogate.")
        self.surrogate = surrogate
        self.model = self.surrogate.model

    @property
    def preferences(self) -> Optional[pd.DataFrame]:
        """Pairwise feedback accumulated by the strategy."""

        return self._preferences

    def _validate_new_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if len(experiments) == 0:
            return pd.DataFrame(columns=[*self.domain.inputs.get_keys(), "labcode"])
        return self.surrogate.validate_pairwise_experiments(experiments)

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
        retrain: bool = True,
        *,
        preferences: Optional[pd.DataFrame] = None,
    ) -> None:
        """Add designs and their pairwise preference feedback.

        Args:
            experiments: New designs with input columns and a unique ``labcode``.
                Pass an empty DataFrame when only adding comparisons between
                designs already known to the strategy.
            preferences: Pairwise feedback with columns ``labcode_A``,
                ``labcode_B``, and ``preference``. A positive sign means A won;
                a negative sign means B won. Zero-valued ties are retained in
                strategy state and ignored by the pairwise surrogate during fit.
            replace: Replace all stored designs and preferences instead of
                appending them.
            retrain: Refit the preference model when sufficient feedback exists.
        """

        if preferences is None:
            raise ValueError(
                "PreferenceStrategy.tell requires a `preferences` DataFrame."
            )
        new_experiments = self._validate_new_experiments(experiments)
        if replace or self.experiments is None:
            combined_experiments = new_experiments.reset_index(drop=True)
        elif new_experiments.empty:
            combined_experiments = self.experiments
        else:
            combined_experiments = pd.concat(
                [self.experiments, new_experiments], ignore_index=True
            )
        if len(combined_experiments) == 0:
            raise ValueError("No preference experiments have been provided.")
        combined_experiments = self.surrogate.validate_pairwise_experiments(
            combined_experiments
        )

        new_preferences = self.surrogate.validate_preferences(
            preferences, combined_experiments
        )
        if replace or self.preferences is None or self.preferences.empty:
            combined_preferences = new_preferences.reset_index(drop=True)
        elif new_preferences.empty:
            combined_preferences = self.preferences
        else:
            combined_preferences = pd.concat(
                [self.preferences, new_preferences], ignore_index=True
            )
        combined_preferences = self.surrogate.validate_preferences(
            combined_preferences, combined_experiments
        )

        self._experiments = combined_experiments
        self._preferences = combined_preferences
        if replace:
            self._is_fitted = False
        if retrain and self.has_sufficient_experiments():
            self.fit()
            self._tell()

    def has_sufficient_experiments(self) -> bool:
        return (
            self.experiments is not None
            and len(self.experiments) >= 2
            and self.preferences is not None
            and (self.preferences["preference"] != 0).any()
        )

    def _validate_fit_experiments(self) -> None:
        if not self.has_sufficient_experiments():
            raise ValueError(
                "At least two designs and one non-tied comparison are required."
            )
        assert self.experiments is not None
        assert self.preferences is not None
        self.surrogate.validate_pairwise_experiments(self.experiments)
        self.surrogate.validate_preferences(self.preferences, self.experiments)

    def _fit(self, experiments: pd.DataFrame) -> None:
        assert self.preferences is not None
        self.surrogate.fit(experiments, self.preferences)
        self.model = self.surrogate.model

    def _predict_objectives(self, predictions: pd.DataFrame) -> pd.DataFrame:
        # Maximization uses fixed bounds and needs no observed utility values.
        output = self.domain.outputs[0]
        assert isinstance(output.objective, MaximizeObjective)
        return pd.DataFrame(
            {f"{output.key}_des": output.objective(predictions[f"{output.key}_pred"])}
        )

    def _get_acqf_experiments(self) -> pd.DataFrame:
        assert self.experiments is not None
        return self.experiments

    def _get_acqfs(self, n: int) -> list[AcquisitionFunction]:
        if not self.is_fitted or self.model is None:
            raise ValueError("Preference model is not fitted.")
        X_train, X_pending = self.get_acqf_input_tensors()
        seed = self._get_seed()
        if isinstance(self.acquisition_function, qEUBO):
            return [
                qExpectedUtilityOfBestOption(
                    pref_model=self.model,
                    objective=IdentityMCObjective(),
                    sampler=SobolQMCNormalSampler(
                        sample_shape=torch.Size(
                            [self.acquisition_function.n_mc_samples]
                        ),
                        seed=seed,
                    ),
                    X_pending=X_pending,
                )
            ]
        params = self.acquisition_function.model_dump()
        return [
            get_acquisition_function(
                self.acquisition_function.__class__.__name__,
                self.model,
                IdentityMCObjective(),
                X_observed=X_train,
                X_pending=X_pending,
                mc_samples=self.acquisition_function.n_mc_samples,
                beta=params.get("beta", 0.2),
                cache_root=None,
                prune_baseline=params.get("prune_baseline", True),
                seed=seed,
            )
        ]

    def _ask(self, candidate_count: Optional[PositiveInt] = None) -> pd.DataFrame:
        default_candidate_count = (
            2 if isinstance(self.acquisition_function, qEUBO) else 1
        )
        candidate_count = (
            default_candidate_count if candidate_count is None else candidate_count
        )
        if isinstance(self.acquisition_function, qEUBO) and candidate_count < 2:
            raise ValueError(
                "PreferenceStrategy requires at least two candidates to form a "
                "comparison batch."
            )
        return super()._ask(candidate_count)

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: AnyPreferenceAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: SurrogateDataModel | None = None,
        seed: int | None = None,
        convergence_criterion: AnyConvergenceCriterion | None = None,
        include_infeasible_exps_in_acqf_calc: bool = False,
        frequency_hyperopt: Literal[0] = 0,
        folds: int = 5,
    ) -> Self:
        """Create a preferential Bayesian optimization strategy."""

        return make_strategy(cls, DataModel, locals())
