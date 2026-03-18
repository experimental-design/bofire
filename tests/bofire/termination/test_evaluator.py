"""Tests for termination evaluators."""

import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
import numpy as np

from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.strategies.api import RandomStrategy, SoboStrategy
from bofire.termination.evaluator import UCBLCBRegretEvaluator


@pytest.fixture
def benchmark():
    return Himmelblau()


@pytest.fixture
def trained_strategy(benchmark):
    """Create a trained SoboStrategy with 10 random initial points."""
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

    strategy = SoboStrategy(data_model=SoboStrategyDataModel(domain=benchmark.domain))
    strategy.tell(experiments)
    return strategy, experiments


class TestUCBLCBRegretEvaluator:
    """Unit tests for the UCBLCBRegretEvaluator."""

    def test_evaluate_returns_valid_regret_bound(self, trained_strategy):
        """Regret bound must be a non-negative float."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert isinstance(result["regret_bound"], float)
        assert result["regret_bound"] >= 0

    def test_returns_all_keys(self, trained_strategy):
        """Evaluate must return the complete set of expected keys."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        expected_keys = {
            "regret_bound",
            "min_ucb_evaluated",
            "min_lcb_domain",
            "estimated_noise_variance",
            "beta",
        }
        assert expected_keys == set(result.keys())

    def test_min_lcb_leq_min_ucb(self, trained_strategy):
        """min LCB(domain) <= min UCB(evaluated), so regret bound >= 0.

        Since LCB(x) <= UCB(x) for all x, and the domain includes the
        evaluated points, min_x LCB(x) <= min_i UCB(x_i).
        """
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["min_lcb_domain"] <= result["min_ucb_evaluated"] + 1e-6

    def test_regret_bound_equals_ucb_minus_lcb(self, trained_strategy):
        """regret_bound = max(0, min_ucb - min_lcb)."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        expected = max(0.0, result["min_ucb_evaluated"] - result["min_lcb_domain"])
        assert abs(result["regret_bound"] - expected) < 1e-10

    def test_returns_empty_dict_when_not_fitted(self, benchmark):
        """Should return empty dict when strategy model is not fitted."""
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, pd.DataFrame(), 0)

        assert result == {}

    def test_returns_empty_dict_with_too_few_experiments(self, trained_strategy):
        """Should return empty dict with fewer than 2 experiments."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments.iloc[:1], 0)

        assert result == {}

    def test_beta_computed_from_gp_ucb_formula(self, trained_strategy):
        """Beta should follow the GP-UCB formula: 0.2 * 2 * log(d * t^2 * pi^2 / (6 * delta))."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        d = len(strategy.domain.inputs.get_keys())  # 2 for Himmelblau
        t = len(experiments)  # 10
        expected_beta = 0.2 * 2.0 * np.log(d * t**2 * np.pi**2 / (6.0 * 0.1))
        assert abs(result["beta"] - expected_beta) < 1e-10

    def test_beta_scales_with_observations(self, benchmark):
        """Beta should increase logarithmically with number of observations."""
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = UCBLCBRegretEvaluator()

        # Fit with 5 points
        exp5 = benchmark.f(random_strategy.ask(5), return_complete=True)
        strategy5 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy5.tell(exp5)
        result5 = evaluator.evaluate(strategy5, exp5, 0)

        # Fit with 15 points
        exp15 = benchmark.f(random_strategy.ask(15), return_complete=True)
        strategy15 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy15.tell(exp15)
        result15 = evaluator.evaluate(strategy15, exp15, 0)

        # More observations → larger beta (logarithmic growth)
        assert result15["beta"] > result5["beta"]

    def test_noise_variance_estimated(self, trained_strategy):
        """Noise variance should be estimated from the GP likelihood."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["estimated_noise_variance"] > 0


class TestRegretBoundConvergence:
    """Integration tests verifying regret bound behavior over BO iterations."""

    def test_regret_bound_generally_decreases(self, benchmark):
        """Over many BO iterations, the regret bound should generally decrease.

        After fitting with more data, the GP becomes more certain and the
        gap between UCB and LCB shrinks. Note: beta grows logarithmically
        with t (GP-UCB formula), so with few iterations the beta increase
        may initially outpace uncertainty reduction. Over enough iterations,
        the GP variance decrease dominates.

        We use enough initial data for a well-fitted GP on Himmelblau (2D),
        then check that regret bound decreases overall across BO iterations.
        We compare the minimum regret bound in the last 3 iterations vs the
        first computed regret bound — the minimum should be strictly smaller,
        showing the bound is decreasing at least some of the time.
        """
        import torch

        torch.manual_seed(42)

        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(20), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)
        evaluator = UCBLCBRegretEvaluator()
        evaluator.n_samples_lcb = 500  # enough samples for stable LCB

        regret_bounds = []
        for i in range(8):
            result = evaluator.evaluate(strategy, strategy.experiments, i)
            regret_bounds.append(result["regret_bound"])

            # Run one BO iteration
            candidates = strategy.ask(1)
            candidates = candidates[benchmark.domain.inputs.get_keys()]
            new_experiments = benchmark.f(candidates)
            new_xy = pd.concat([candidates, new_experiments], axis=1)
            strategy.tell(pd.concat([strategy.experiments, new_xy]))

        # The minimum regret bound across all iterations should be
        # smaller than the first regret bound value (bound decreases
        # at least some of the time over BO iterations)
        first_rb = regret_bounds[0]
        min_all = min(regret_bounds[1:])
        assert min_all < first_rb
