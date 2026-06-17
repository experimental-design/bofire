"""Shared settings for nonlinear constraint strategy tests."""

from bofire.data_models.strategies.api import BotorchOptimizer


# Matches docs/tutorials/advanced_examples/nonlinear_constraints_maximizing_yield.py
NONLINEAR_BOTORCH_OPTIMIZER = BotorchOptimizer(
    n_restarts=2,
    n_raw_samples=64,
    batch_limit=1,
)
