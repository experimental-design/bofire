from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.sampling.normal import SobolQMCNormalSampler
from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import qEUBO, qLogNEI
from bofire.data_models.api import Domain
from bofire.data_models.strategies.api import PreferenceStrategy as DataModel
from bofire.data_models.strategies.convergence_criteria.api import (
    AnyConvergenceCriterion,
)
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.surrogates.api import PairwiseGPSurrogate as SurrogateDataModel
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.predictives.acqf_optimization import (
    AcquisitionOptimizer,
    get_optimizer,
)
from bofire.strategies.predictives.predictive import PredictiveStrategy
from bofire.strategies.strategy import make_strategy
from bofire.surrogates.mapper import map as map_surrogate
from bofire.surrogates.pairwise_gp import PairwiseGPSurrogate
from bofire.utils.torch_tools import tkwargs


class PreferenceStrategy(PredictiveStrategy):
    """Preferential Bayesian optimization using a pairwise GP."""

    PREFERENCE_COLUMNS = ("labcode_A", "labcode_B", "preference")

    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self.acqf_optimizer: AcquisitionOptimizer = get_optimizer(
            data_model.acquisition_optimizer
        )
        assert data_model.surrogate_spec is not None
        surrogate = map_surrogate(data_model.surrogate_spec)
        if not isinstance(surrogate, PairwiseGPSurrogate):
            raise TypeError("PreferenceStrategy requires a PairwiseGPSurrogate.")
        self.surrogate = surrogate
        self.model = self.surrogate.model
        self._preferences: Optional[pd.DataFrame] = None
        torch.manual_seed(self.seed)

    @property
    def preferences(self) -> Optional[pd.DataFrame]:
        """Pairwise feedback accumulated by the strategy."""

        return self._preferences

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return self.surrogate.input_preprocessing_specs

    def _validate_new_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        experiments = experiments.copy()
        if len(experiments) == 0:
            return pd.DataFrame(columns=[*self.domain.inputs.get_keys(), "labcode"])
        experiments = self.domain.inputs.validate_experiments(experiments, strict=False)
        if "labcode" not in experiments.columns:
            raise ValueError(
                "PreferenceStrategy experiments require a 'labcode' column."
            )
        if experiments["labcode"].isna().any():
            raise ValueError("PreferenceStrategy labcodes must not be missing.")
        if experiments["labcode"].duplicated().any():
            duplicates = sorted(
                experiments.loc[
                    experiments["labcode"].duplicated(keep=False), "labcode"
                ]
                .unique()
                .tolist()
            )
            raise ValueError(f"Duplicate labcodes in experiments: {duplicates}.")
        return experiments[[*self.domain.inputs.get_keys(), "labcode"]]

    def _validate_preferences(
        self, preferences: pd.DataFrame, experiments: pd.DataFrame
    ) -> pd.DataFrame:
        preferences = preferences.copy()
        if len(preferences) == 0:
            return pd.DataFrame(columns=self.PREFERENCE_COLUMNS)
        missing = set(self.PREFERENCE_COLUMNS) - set(preferences.columns)
        if missing:
            raise ValueError(
                f"`preferences` is missing required columns: {sorted(missing)}."
            )
        preferences = preferences[list(self.PREFERENCE_COLUMNS)]
        preferences["preference"] = pd.to_numeric(
            preferences["preference"], errors="raise"
        )
        if not np.isfinite(preferences["preference"].to_numpy()).all():
            raise ValueError("Preference values must be finite.")
        if preferences[["labcode_A", "labcode_B"]].isna().any().any():
            raise ValueError("Preference labcodes must not be missing.")
        if (preferences["labcode_A"] == preferences["labcode_B"]).any():
            raise ValueError("A design cannot be compared with itself.")

        known_labcodes = set(experiments["labcode"].tolist())
        referenced_labcodes = set(preferences["labcode_A"].tolist()) | set(
            preferences["labcode_B"].tolist()
        )
        unknown = referenced_labcodes - known_labcodes
        if unknown:
            raise ValueError(
                f"`preferences` references unknown labcodes: {sorted(unknown)}."
            )
        return preferences

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
        self.tell_preferences(
            experiments=experiments,
            preferences=preferences,
            replace=replace,
            retrain=retrain,
        )

    def tell_preferences(
        self,
        experiments: pd.DataFrame,
        preferences: pd.DataFrame,
        replace: bool = False,
        retrain: bool = True,
    ) -> None:
        """Add pairwise observations through an explicit preference API."""

        new_experiments = self._validate_new_experiments(experiments)
        if replace or self.experiments is None:
            combined_experiments = new_experiments.reset_index(drop=True)
        else:
            combined_experiments = pd.concat(
                [self.experiments, new_experiments], ignore_index=True
            )
        if len(combined_experiments) == 0:
            raise ValueError("No preference experiments have been provided.")
        if combined_experiments["labcode"].duplicated().any():
            duplicates = sorted(
                combined_experiments.loc[
                    combined_experiments["labcode"].duplicated(keep=False), "labcode"
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                "Labcodes must remain unique when appending experiments; "
                f"duplicates: {duplicates}."
            )

        new_preferences = self._validate_preferences(preferences, combined_experiments)
        if replace or self.preferences is None:
            combined_preferences = new_preferences.reset_index(drop=True)
        else:
            combined_preferences = pd.concat(
                [self.preferences, new_preferences], ignore_index=True
            )
        combined_preferences = self._validate_preferences(
            combined_preferences, combined_experiments
        )

        self._experiments = combined_experiments
        self._preferences = combined_preferences
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

    def fit(self) -> None:
        if not self.has_sufficient_experiments():
            raise ValueError(
                "At least two designs and one non-tied comparison are required."
            )
        assert self.experiments is not None
        assert self.preferences is not None
        self._fit(self.experiments)
        self._is_fitted = True

    def _fit(self, experiments: pd.DataFrame) -> None:
        assert self.preferences is not None
        self.surrogate.fit(experiments, self.preferences)
        self.model = self.surrogate.model

    def _predict(self, experiments: pd.DataFrame):
        return self.surrogate._predict(experiments)

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Predict latent utility without requiring an observed utility column."""

        if not self.is_fitted:
            raise ValueError("Model not yet fitted.")
        predictions = self.surrogate.predict(experiments)
        predictions.index = experiments.index
        utility_key = self.domain.outputs[0].key
        # The generic PredictiveStrategy adapts objectives using observed output
        # values. Preferential BO has no observed latent utilities, so use the
        # posterior location itself as the adaptation frame. This is immaterial
        # for the required MaximizeObjective but preserves the standard `_des`
        # candidate column contract.
        adaptation = pd.DataFrame(
            {utility_key: predictions[f"{utility_key}_pred"]},
            index=predictions.index,
        )
        objectives = self.domain.outputs(
            predictions, experiments_adapt=adaptation, predictions=True
        )
        return pd.concat([predictions, objectives], axis=1)

    def _get_acqf(self) -> AcquisitionFunction:
        if not self.is_fitted or self.model is None:
            raise ValueError("Preference model is not fitted.")

        X_pending = None
        if self.candidates is not None and len(self.candidates) > 0:
            transformed = self.domain.inputs.transform(
                self.candidates, self.input_preprocessing_specs
            )
            X_pending = torch.from_numpy(transformed.to_numpy(dtype=float)).to(
                **tkwargs
            )

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.acquisition_function.n_mc_samples]),
            seed=self._get_seed(),
        )
        if isinstance(self.acquisition_function, qEUBO):
            return qExpectedUtilityOfBestOption(
                pref_model=self.model,
                sampler=sampler,
                X_pending=X_pending,
            )

        if self.experiments is None:
            raise ValueError("No preference experiments have been provided.")
        transformed_baseline = self.domain.inputs.transform(
            self.experiments, self.input_preprocessing_specs
        )
        X_baseline = torch.from_numpy(transformed_baseline.to_numpy(dtype=float)).to(
            **tkwargs
        )
        return qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=X_baseline,
            sampler=sampler,
            X_pending=X_pending,
            prune_baseline=self.acquisition_function.prune_baseline,
        )

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
        return self.acqf_optimizer.optimize(
            candidate_count=candidate_count,
            acqfs=[self._get_acqf()],
            domain=self.domain,
            experiments=self.experiments,
        )

    def calc_acquisition(
        self, candidates: pd.DataFrame, combined: bool = False
    ) -> np.ndarray:
        transformed = self.domain.inputs.transform(
            candidates, self.input_preprocessing_specs
        )
        X = torch.from_numpy(transformed.to_numpy(dtype=float)).to(**tkwargs)
        if not combined:
            X = X.unsqueeze(-2)
        with torch.no_grad():
            return self._get_acqf()(X).cpu().detach().numpy()

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: qEUBO | qLogNEI | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_spec: SurrogateDataModel | None = None,
        seed: int | None = None,
        convergence_criterion: AnyConvergenceCriterion | None = None,
    ) -> Self:
        """Create a preferential Bayesian optimization strategy."""

        return make_strategy(cls, DataModel, locals())
