"""Tests for timeseries utilities."""

import pandas as pd
import pytest

from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    MolecularInput,
)
from bofire.utils.timeseries import infer_trajectory_id


def test_infer_trajectory_id_continuous_only():
    """Test trajectory ID inference with continuous features only."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="temperature", bounds=(20, 80)),
            ContinuousInput(key="pressure", bounds=(1, 5)),
        ]
    )

    # Create experiments with 3 distinct trajectories
    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20, 0, 10, 20],
            "temperature": [25, 25, 25, 30, 30, 30, 25, 25, 25],
            "pressure": [2, 2, 2, 2, 2, 2, 3, 3, 3],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
            "valid_yield": [1] * 9,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # Check that we have 3 trajectories
    assert len(trajectory_ids.unique()) == 3

    # Check that rows with same temp/pressure get same ID
    assert trajectory_ids.iloc[0] == trajectory_ids.iloc[1] == trajectory_ids.iloc[2]
    assert trajectory_ids.iloc[3] == trajectory_ids.iloc[4] == trajectory_ids.iloc[5]
    assert trajectory_ids.iloc[6] == trajectory_ids.iloc[7] == trajectory_ids.iloc[8]

    # All three trajectories should have different IDs
    assert (
        len({trajectory_ids.iloc[0], trajectory_ids.iloc[3], trajectory_ids.iloc[6]})
        == 3
    )


def test_infer_trajectory_id_with_categorical():
    """Test trajectory ID inference with categorical features."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            CategoricalInput(key="catalyst", categories=["A", "B", "C"]),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20, 0, 10, 20],
            "catalyst": ["A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "temperature": [25, 25, 25, 25, 25, 25, 30, 30, 30],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
            "valid_yield": [1] * 9,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # Should have 3 trajectories: (A, 25), (B, 25), (A, 30)
    assert len(trajectory_ids.unique()) == 3

    # Rows 0-2: catalyst A, temp 25
    assert trajectory_ids.iloc[0] == trajectory_ids.iloc[1] == trajectory_ids.iloc[2]

    # Rows 3-5: catalyst B, temp 25
    assert trajectory_ids.iloc[3] == trajectory_ids.iloc[4] == trajectory_ids.iloc[5]

    # Rows 6-8: catalyst A, temp 30
    assert trajectory_ids.iloc[6] == trajectory_ids.iloc[7] == trajectory_ids.iloc[8]

    # All three should be different
    assert (
        len({trajectory_ids.iloc[0], trajectory_ids.iloc[3], trajectory_ids.iloc[6]})
        == 3
    )


def test_infer_trajectory_id_with_discrete():
    """Test trajectory ID inference with discrete features."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            DiscreteInput(key="n_cycles", values=[1, 2, 3, 4, 5]),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20],
            "n_cycles": [2, 2, 2, 3, 3, 3],
            "temperature": [25, 25, 25, 25, 25, 25],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
            "valid_yield": [1] * 6,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # Should have 2 trajectories: n_cycles=2 and n_cycles=3
    assert len(trajectory_ids.unique()) == 2

    # First 3 rows should have same ID
    assert trajectory_ids.iloc[0] == trajectory_ids.iloc[1] == trajectory_ids.iloc[2]

    # Last 3 rows should have same ID
    assert trajectory_ids.iloc[3] == trajectory_ids.iloc[4] == trajectory_ids.iloc[5]

    # The two groups should be different
    assert trajectory_ids.iloc[0] != trajectory_ids.iloc[3]


def test_infer_trajectory_id_mixed_features():
    """Test trajectory ID inference with mixed feature types."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            CategoricalInput(
                key="solvent", categories=["water", "ethanol", "methanol"]
            ),
            DiscreteInput(key="stirring_speed", values=[100, 200, 300]),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20, 0, 10, 20],
            "solvent": [
                "water",
                "water",
                "water",
                "ethanol",
                "ethanol",
                "ethanol",
                "water",
                "water",
                "water",
            ],
            "stirring_speed": [100, 100, 100, 100, 100, 100, 200, 200, 200],
            "temperature": [25, 25, 25, 25, 25, 25, 25, 25, 25],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
            "valid_yield": [1] * 9,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # Should have 3 trajectories
    assert len(trajectory_ids.unique()) == 3

    # Check grouping
    assert trajectory_ids.iloc[0] == trajectory_ids.iloc[1] == trajectory_ids.iloc[2]
    assert trajectory_ids.iloc[3] == trajectory_ids.iloc[4] == trajectory_ids.iloc[5]
    assert trajectory_ids.iloc[6] == trajectory_ids.iloc[7] == trajectory_ids.iloc[8]


def test_infer_trajectory_id_with_eps():
    """Test that eps parameter works for continuous features."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    # Create experiments with small differences in temperature
    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20],
            "temperature": [25.0, 25.0, 25.0, 25.0000001, 25.0000001, 25.0000001],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
            "valid_yield": [1] * 6,
        }
    )

    # With default eps=1e-6, should group together
    trajectory_ids_tight = infer_trajectory_id(experiments, inputs, eps=1e-6)
    assert len(trajectory_ids_tight.unique()) == 1

    # With stricter eps, should be separate
    trajectory_ids_strict = infer_trajectory_id(experiments, inputs, eps=1e-8)
    assert len(trajectory_ids_strict.unique()) == 2


def test_infer_trajectory_id_only_timeseries():
    """Test when only timeseries feature exists (edge case)."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 30],
            "yield": [0.1, 0.2, 0.3, 0.4],
            "valid_yield": [1] * 4,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # All rows should have same trajectory ID (only one trajectory)
    assert len(trajectory_ids.unique()) == 1
    assert trajectory_ids.iloc[0] == 0


def test_infer_trajectory_id_no_timeseries_error():
    """Test that error is raised when no timeseries feature exists."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="temperature", bounds=(20, 80)),
            ContinuousInput(key="pressure", bounds=(1, 5)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "temperature": [25, 30],
            "pressure": [2, 3],
            "yield": [0.1, 0.2],
            "valid_yield": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="No timeseries feature found"):
        infer_trajectory_id(experiments, inputs)


def test_infer_trajectory_id_missing_columns_error():
    """Test that error is raised when required columns are missing."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20],
            # Missing temperature column
            "yield": [0.1, 0.2, 0.3],
            "valid_yield": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="Required input feature columns missing"):
        infer_trajectory_id(experiments, inputs)


def test_domain_add_trajectory_id():
    """Test the Domain.add_trajectory_id convenience method."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="yield")])
    domain = Domain(inputs=inputs, outputs=outputs)

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20],
            "temperature": [25, 25, 25, 30, 30, 30],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
            "valid_yield": [1] * 6,
        }
    )

    # Test that add_trajectory_id works
    result = domain.add_trajectory_id(experiments)

    # Check that _trajectory_id was added
    assert "_trajectory_id" in result.columns

    # Check that original dataframe was not modified
    assert "_trajectory_id" not in experiments.columns

    # Check grouping is correct
    assert len(result["_trajectory_id"].unique()) == 2
    assert result["_trajectory_id"].iloc[0] == result["_trajectory_id"].iloc[1]
    assert result["_trajectory_id"].iloc[3] == result["_trajectory_id"].iloc[4]
    assert result["_trajectory_id"].iloc[0] != result["_trajectory_id"].iloc[3]


def test_infer_trajectory_id_with_molecular():
    """Test trajectory ID inference with molecular features."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            MolecularInput(key="solvent"),
            ContinuousInput(key="temperature", bounds=(20, 80)),
        ]
    )

    experiments = pd.DataFrame(
        {
            "time": [0, 10, 20, 0, 10, 20, 0, 10, 20],
            "solvent": [
                "O",
                "O",
                "O",
                "CCO",
                "CCO",
                "CCO",
                "O",
                "O",
                "O",
            ],  # water, ethanol, water
            "temperature": [25, 25, 25, 25, 25, 25, 30, 30, 30],
            "yield": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
            "valid_yield": [1] * 9,
        }
    )

    trajectory_ids = infer_trajectory_id(experiments, inputs)

    # Should have 3 trajectories: (water, 25), (ethanol, 25), (water, 30)
    assert len(trajectory_ids.unique()) == 3

    # Check grouping
    assert trajectory_ids.iloc[0] == trajectory_ids.iloc[1] == trajectory_ids.iloc[2]
    assert trajectory_ids.iloc[3] == trajectory_ids.iloc[4] == trajectory_ids.iloc[5]
    assert trajectory_ids.iloc[6] == trajectory_ids.iloc[7] == trajectory_ids.iloc[8]
