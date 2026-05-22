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
                ``labcode_B``, ``preference``. ``preference`` is a float in
                ``[-1, 1]``; sign determines the winner (``>0`` = A wins,
                ``<0`` = B wins), magnitude is currently discarded, ``==0``
                rows are dropped as ties.
            options: Additional keyword arguments forwarded to ``_fit_pairwise``.
        """
        # validate experiment inputs (skip outputs validation: pairwise GP has
        # no observed Y values in the experiments DataFrame).
        experiments = self.inputs.validate_experiments(experiments, strict=False)

        if "labcode" not in experiments.columns:
            raise ValueError(
                "PairwiseGPSurrogate requires a 'labcode' column on experiments "
                "to reference preferences."
            )

        if experiments["labcode"].duplicated().any():
            dups = (
                experiments.loc[
                    experiments["labcode"].duplicated(keep=False), "labcode"
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Duplicate labcodes in experiments: {sorted(dups)}. "
                "PairwiseGPSurrogate requires unique labcodes."
            )

        expected_cols = set(self.PREFERENCE_COLUMNS)
        missing = expected_cols - set(preferences.columns)
        if missing:
            raise ValueError(
                f"`preferences` is missing required columns: {sorted(missing)}. "
                f"Expected at least {sorted(expected_cols)}."
            )

        valid_labcodes = set(experiments["labcode"].tolist())
        ref_labcodes = set(preferences["labcode_A"].tolist()) | set(
            preferences["labcode_B"].tolist()
        )
        unknown = ref_labcodes - valid_labcodes
        if unknown:
            raise ValueError(
                "`preferences` references labcodes not present in experiments: "
                f"{sorted(unknown)}."
            )

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
