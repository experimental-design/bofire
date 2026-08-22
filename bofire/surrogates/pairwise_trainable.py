import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import InputTransform

from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.surrogates.scaler import AnyScaler
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.utils import get_input_transform
from bofire.utils.torch_tools import tkwargs


class PairwiseTrainableSurrogate(ABC):
    """Mixin for surrogates that train on pairwise preference data.

    Structurally parallel to :class:`bofire.surrogates.trainable.TrainableSurrogate`
    but with a different fit signature: ``fit(experiments, preferences)`` instead
    of ``fit(experiments)``. The ``preferences`` DataFrame carries the label
    signal; ``experiments`` provides the candidate designs referenced by
    ``labcode``.
    """

    # These attributes are provided by Surrogate / BotorchSurrogate via
    # multiple inheritance.
    inputs: Inputs
    outputs: Outputs
    predict: Callable[..., pd.DataFrame]
    input_preprocessing_specs: InputTransformSpecs
    categorical_encodings: InputTransformSpecs
    scaler: AnyScaler
    engineered_features: EngineeredFeatures

    PREFERENCE_COLUMNS = ("labcode_A", "labcode_B", "preference")

    def validate_pairwise_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Validate designs referenced by pairwise preference observations."""
        experiments = self.inputs.validate_experiments(experiments.copy(), strict=False)
        if "labcode" not in experiments.columns:
            raise ValueError(
                "Pairwise preference experiments require a 'labcode' column."
            )
        if experiments["labcode"].isna().any():
            raise ValueError("Pairwise preference labcodes must not be missing.")
        if experiments["labcode"].duplicated().any():
            duplicates = sorted(
                experiments.loc[
                    experiments["labcode"].duplicated(keep=False), "labcode"
                ]
                .unique()
                .tolist()
            )
            raise ValueError(f"Duplicate labcodes in experiments: {duplicates}.")
        return experiments[[*self.inputs.get_keys(), "labcode"]]

    def validate_preferences(
        self, preferences: pd.DataFrame, experiments: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate pairwise labels against their referenced experiments."""
        preferences = preferences.copy()
        if len(preferences) == 0:
            return pd.DataFrame(columns=self.PREFERENCE_COLUMNS)

        missing = set(self.PREFERENCE_COLUMNS) - set(preferences.columns)
        if missing:
            raise ValueError(
                f"`preferences` is missing required columns: {sorted(missing)}. "
                f"Expected at least {sorted(self.PREFERENCE_COLUMNS)}."
            )
        preferences = preferences[list(self.PREFERENCE_COLUMNS)]
        preferences["preference"] = pd.to_numeric(
            preferences["preference"], errors="raise"
        )
        if not preferences["preference"].isin((-1.0, 0.0, 1.0)).all():
            raise ValueError("Preference values must be one of -1, 0, or 1.")
        if preferences[["labcode_A", "labcode_B"]].isna().any().any():
            raise ValueError("Preference labcodes must not be missing.")
        if (preferences["labcode_A"] == preferences["labcode_B"]).any():
            raise ValueError("A design cannot be compared with itself.")

        valid_labcodes = set(experiments["labcode"].tolist())
        referenced_labcodes = set(preferences["labcode_A"].tolist()) | set(
            preferences["labcode_B"].tolist()
        )
        unknown = referenced_labcodes - valid_labcodes
        if unknown:
            raise ValueError(
                "`preferences` references unknown labcodes not present in "
                f"experiments: {sorted(unknown)}."
            )
        return preferences

    def fit(
        self,
        experiments: pd.DataFrame,
        preferences: pd.DataFrame,
        options: Optional[Dict] = None,
    ):
        """Fit the pairwise surrogate to preference data.

        Args:
            experiments: DataFrame with input columns plus a ``labcode`` column.
                Output columns are ignored if present.
            preferences: DataFrame with exactly the columns ``labcode_A``,
                ``labcode_B``, ``preference``. ``preference`` must be ``1``
                when A wins, ``-1`` when B wins, or ``0`` for a tie. Tie rows
                are dropped.
            options: Additional keyword arguments forwarded to ``_fit_pairwise``.
        """
        # Skip outputs validation: the latent utility has no observed Y values.
        experiments = self.validate_pairwise_experiments(experiments)
        preferences = self.validate_preferences(preferences, experiments)

        # sign conversion: drop ties (preference == 0)
        pref_values = preferences["preference"].astype(float)
        tie_mask = pref_values == 0.0
        n_ties = int(tie_mask.sum())
        if n_ties > 0:
            warnings.warn(
                f"Dropping {n_ties} pair(s) with preference == 0 (ties).",
                stacklevel=2,
            )
        preferences = preferences.loc[~tie_mask].reset_index(drop=True)

        if len(preferences) == 0:
            raise ValueError("No valid pairs remain after dropping ties.")

        # build idx_map: labcode -> position in datapoints tensor
        idx_map = {
            labcode: i for i, labcode in enumerate(experiments["labcode"].tolist())
        }

        # winner/loser indices from sign of preference
        pref_signs = preferences["preference"].astype(float).to_numpy()
        labcode_A = preferences["labcode_A"].to_numpy()
        labcode_B = preferences["labcode_B"].to_numpy()
        winners = np.where(pref_signs > 0, labcode_A, labcode_B)
        losers = np.where(pref_signs > 0, labcode_B, labcode_A)
        winner_idx = np.array([idx_map[w] for w in winners], dtype=np.int64)
        loser_idx = np.array([idx_map[loser] for loser in losers], dtype=np.int64)
        comparisons = torch.from_numpy(np.stack([winner_idx, loser_idx], axis=1)).to(
            dtype=torch.long
        )

        # datapoints tensor (float64), pre-transformed via BoFire's
        # categorical preprocessing; BoTorch's input_transform is applied
        # internally by the model.
        X = experiments[self.inputs.get_keys()]
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        datapoints = torch.from_numpy(transformed_X.values).to(**tkwargs)

        input_transform = get_input_transform(
            inputs=self.inputs,
            engineered_features=self.engineered_features,
            scaler_type=self.scaler,
            categorical_encodings=self.categorical_encodings,
            X=X,
        )

        options = options or {}
        self._fit_pairwise(
            datapoints=datapoints,
            comparisons=comparisons,
            input_transform=input_transform,
            **options,
        )

    @abstractmethod
    def _fit_pairwise(
        self,
        datapoints: torch.Tensor,
        comparisons: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        **kwargs,
    ):
        """Fit the underlying pairwise model.

        Args:
            datapoints: Unique candidate features, shape ``(n, d)``, float64.
            comparisons: Long tensor of shape ``(m, 2)`` where each row is
                ``[winner_idx, loser_idx]`` into ``datapoints``.
            input_transform: Optional BoTorch input transform to attach to
                the underlying model.
        """
