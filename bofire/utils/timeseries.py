"""Utilities for working with timeseries data in BoFire."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from bofire.data_models.domain.features import Inputs


def infer_trajectory_id(
    experiments: pd.DataFrame,
    inputs: "Inputs",
    eps: float = 1e-6,
) -> pd.Series:
    """
    Automatically infer trajectory IDs by grouping experiments with the same
    non-timeseries input feature values.

    For each row in the experiments DataFrame, this function identifies which
    trajectory it belongs to by comparing the values of all non-timeseries input
    features. Rows with the same (or nearly same, within eps) values for all
    non-timeseries features are assigned the same trajectory ID.

    This is useful when you have experimental data from multiple runs/trajectories
    where the timeseries feature varies, but you haven't manually labeled which
    observations belong to which trajectory.

    Args:
        experiments: DataFrame with experimental data. Must contain columns for
            all input features defined in inputs.
        inputs: Inputs object that defines the input features and identifies
            which feature (if any) is marked as timeseries.
        eps: Tolerance for comparing continuous values. Two continuous values
            are considered equal if their absolute difference is less than eps.
            Default: 1e-6. Does not apply to discrete, categorical, or molecular
            features, which use exact equality.

    Returns:
        pd.Series: Series of integer trajectory IDs (0, 1, 2, ...) corresponding
            to each row in experiments. Rows with the same trajectory ID belong
            to the same experimental run/trajectory.

    Raises:
        ValueError: If no timeseries feature is found in inputs.
        ValueError: If required input feature columns are missing from experiments.
        ValueError: If no valid trajectories are found (each row is its own trajectory).

    Example:
        >>> from bofire.data_models.domain.api import Inputs
        >>> from bofire.data_models.features.api import ContinuousInput
        >>> from bofire.utils.timeseries import infer_trajectory_id
        >>> import pandas as pd
        >>>
        >>> # Define inputs with timeseries
        >>> inputs = Inputs(features=[
        ...     ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
        ...     ContinuousInput(key="temperature", bounds=(20, 80)),
        ... ])
        >>>
        >>> # Create experiments (3 trajectories, temperature varies between them)
        >>> experiments = pd.DataFrame({
        ...     'time': [0, 10, 20, 0, 10, 20, 0, 10, 20],
        ...     'temperature': [25, 25, 25, 30, 30, 30, 25, 25, 25],
        ... })
        >>>
        >>> # Infer trajectory IDs
        >>> experiments['_trajectory_id'] = infer_trajectory_id(experiments, inputs)
        >>> print(experiments['_trajectory_id'].tolist())
        [0, 0, 0, 1, 1, 1, 0, 0, 0]  # Rows with temp=25 get same ID
    """
    from bofire.data_models.features.categorical import CategoricalInput
    from bofire.data_models.features.discrete import DiscreteInput
    from bofire.data_models.features.molecular import MolecularInput
    from bofire.data_models.features.numerical import NumericalInput

    # Identify timeseries feature
    timeseries_features = [
        f
        for f in inputs
        if isinstance(f, NumericalInput) and getattr(f, "is_timeseries", False)
    ]

    if len(timeseries_features) == 0:
        raise ValueError(
            "No timeseries feature found in the domain. "
            "At least one input feature must be marked with is_timeseries=True."
        )

    timeseries_key = timeseries_features[0].key

    # Get all non-timeseries input features
    grouping_features = [f for f in inputs if f.key != timeseries_key]

    if len(grouping_features) == 0:
        # Special case: only timeseries feature exists
        # All rows belong to the same trajectory
        return pd.Series(0, index=experiments.index)

    # Check that all required columns are present
    missing_cols = [
        f.key for f in grouping_features if f.key not in experiments.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Required input feature columns missing from experiments: {missing_cols}"
        )

    # Build a grouping key for each row based on non-timeseries features
    # We'll create a tuple for each row representing its grouping values

    def make_grouping_key(row):
        """Create a hashable grouping key for a row."""
        key_parts = []
        for feature in grouping_features:
            value = row[feature.key]

            if isinstance(feature, (CategoricalInput, DiscreteInput, MolecularInput)):
                # Categorical, discrete, and molecular features: use exact value
                key_parts.append(value)
            elif isinstance(feature, NumericalInput):
                # Continuous features: round to tolerance for grouping
                # This handles floating point comparison issues
                if pd.isna(value):
                    key_parts.append(None)
                else:
                    # Round to appropriate decimal places based on eps
                    decimal_places = max(0, int(-np.log10(eps)))
                    rounded_value = round(float(value), decimal_places)
                    key_parts.append(rounded_value)
            else:
                # Fallback: use exact value
                key_parts.append(value)

        return tuple(key_parts)

    # Create grouping keys for all rows
    grouping_keys = experiments.apply(make_grouping_key, axis=1)

    # Map unique grouping keys to trajectory IDs
    unique_keys = grouping_keys.unique()
    key_to_id = {key: idx for idx, key in enumerate(unique_keys)}

    # Assign trajectory IDs
    trajectory_ids = pd.Series(
        [key_to_id[key] for key in grouping_keys],
        index=experiments.index,
    )

    # Validate that actual trajectories were found (not every row is its own trajectory)
    n_trajectories = len(unique_keys)
    n_rows = len(experiments)
    if n_trajectories == n_rows:
        raise ValueError(
            f"No valid timeseries trajectories found. Each row was assigned its own trajectory ID "
            f"({n_trajectories} trajectories for {n_rows} rows), indicating that all "
            f"non-timeseries input features have unique values. For timeseries cross-validation, "
            f"multiple rows must share the same trajectory. Ensure that non-timeseries input "
            f"features have repeated values across time points within each trajectory."
        )

    return trajectory_ids
