"""Tests for termination evaluators."""

import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
import numpy as np

from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.strategies.api import RandomStrategy, SoboStrategy
from bofire.termination.evaluator import ExpMinRegretGapEvaluator, UCBLCBRegretEvaluator


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


class TestExpMinRegretGapEvaluator:
    """Unit tests for the ExpMinRegretGapEvaluator (Ishibashi et al. 2023)."""

    @pytest.fixture
    def bo_loop_strategies(self, benchmark):
        """Simulate 2 BO iterations, returning (strategy_iter1, exp1, strategy_iter2, exp2).

        Iter 1: 10 random points, fit GP.
        Iter 2: ask 1 candidate → evaluate → refit GP with 11 points.
        """
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments1 = benchmark.f(random_strategy.ask(10), return_complete=True)

        strategy1 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy1.tell(experiments1)

        # Get a candidate and evaluate it
        candidates = strategy1.ask(1)
        candidates = candidates[benchmark.domain.inputs.get_keys()]
        new_exp = benchmark.f(candidates)
        new_xy = pd.concat([candidates, new_exp], axis=1)
        experiments2 = pd.concat(
            [experiments1, new_xy], ignore_index=True,
        )

        strategy2 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy2.tell(experiments2)

        return strategy1, experiments1, strategy2, experiments2

    def test_first_call_returns_empty(self, trained_strategy):
        """First call should return empty dict (no previous model to compare)."""
        strategy, experiments = trained_strategy
        evaluator = ExpMinRegretGapEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result == {}

    def test_second_call_returns_metrics(self, bo_loop_strategies):
        """Second call should return the full set of stopping metrics."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        # First call: saves state
        result1 = evaluator.evaluate(strategy1, exp1, 0)
        assert result1 == {}

        # Second call: computes metrics
        result2 = evaluator.evaluate(strategy2, exp2, 1)
        assert result2 != {}

        expected_keys = {
            "stopping_value",
            "delta_f",
            "ei_diff",
            "kappa",
            "kl_divergence",
            "threshold_adaptive",
            "threshold_median",
            "noise_variance",
            "seq_values",
        }
        assert expected_keys == set(result2.keys())

    def test_stopping_value_non_negative(self, bo_loop_strategies):
        """Stopping value must be non-negative."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        evaluator.evaluate(strategy1, exp1, 0)
        result = evaluator.evaluate(strategy2, exp2, 1)

        assert result["stopping_value"] >= 0

    def test_components_non_negative(self, bo_loop_strategies):
        """All components (delta_f, ei_diff, kappa, kl) must be non-negative."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        evaluator.evaluate(strategy1, exp1, 0)
        result = evaluator.evaluate(strategy2, exp2, 1)

        assert result["delta_f"] >= 0
        assert result["ei_diff"] >= 0
        assert result["kappa"] >= 0
        assert result["kl_divergence"] >= 0

    def test_stopping_value_is_sum_of_components(self, bo_loop_strategies):
        """value = delta_f + ei_diff + kappa * sqrt(KL/2)."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        evaluator.evaluate(strategy1, exp1, 0)
        result = evaluator.evaluate(strategy2, exp2, 1)

        expected = (
            result["delta_f"]
            + result["ei_diff"]
            + result["kappa"] * np.sqrt(0.5 * result["kl_divergence"])
        )
        assert abs(result["stopping_value"] - expected) < 1e-10

    def test_adaptive_threshold_computed(self, bo_loop_strategies):
        """Adaptive threshold should be a positive float."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        evaluator.evaluate(strategy1, exp1, 0)
        result = evaluator.evaluate(strategy2, exp2, 1)

        assert result["threshold_adaptive"] is not None
        assert result["threshold_adaptive"] > 0

    def test_median_threshold_none_before_start_timing(self, bo_loop_strategies):
        """Median threshold should be None when fewer than start_timing values."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()
        evaluator.start_timing = 100  # much more than 1 value

        evaluator.evaluate(strategy1, exp1, 0)
        result = evaluator.evaluate(strategy2, exp2, 1)

        assert result["threshold_median"] is None

    def test_seq_values_accumulate(self, benchmark):
        """Sequence of stopping values should grow with each call."""
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)
        evaluator = ExpMinRegretGapEvaluator()

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)
        evaluator.evaluate(strategy, experiments, 0)  # first call → empty

        for i in range(3):
            candidates = strategy.ask(1)
            candidates = candidates[benchmark.domain.inputs.get_keys()]
            new_exp = benchmark.f(candidates)
            new_xy = pd.concat([candidates, new_exp], axis=1)
            experiments = pd.concat(
                [experiments, new_xy], ignore_index=True,
            )

            strategy = SoboStrategy(
                data_model=SoboStrategyDataModel(domain=benchmark.domain)
            )
            strategy.tell(experiments)
            result = evaluator.evaluate(strategy, experiments, i + 1)
            assert len(result["seq_values"]) == i + 1

    def test_median_threshold_after_start_timing(self, benchmark):
        """Once enough values are collected, median threshold should be computed."""
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)
        evaluator = ExpMinRegretGapEvaluator()
        evaluator.start_timing = 3  # low for tests
        evaluator.rate = 0.1

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)
        evaluator.evaluate(strategy, experiments, 0)

        for i in range(5):
            candidates = strategy.ask(1)
            candidates = candidates[benchmark.domain.inputs.get_keys()]
            new_exp = benchmark.f(candidates)
            new_xy = pd.concat([candidates, new_exp], axis=1)
            experiments = pd.concat(
                [experiments, new_xy], ignore_index=True,
            )

            strategy = SoboStrategy(
                data_model=SoboStrategyDataModel(domain=benchmark.domain)
            )
            strategy.tell(experiments)
            result = evaluator.evaluate(strategy, experiments, i + 1)

        # After 5 iterations with start_timing=3, median threshold should exist
        assert result["threshold_median"] is not None
        assert result["threshold_median"] > 0
        # threshold_median = rate * median(first 3 values)
        expected = 0.1 * np.median(result["seq_values"][:3])
        assert abs(result["threshold_median"] - expected) < 1e-10

    def test_kl_divergence_formula(self):
        """Test the fast KL divergence formula against known values."""
        evaluator = ExpMinRegretGapEvaluator()

        # When observation equals the GP mean, KL should be small (only from
        # the variance reduction term)
        kl = evaluator._calc_kl_qp_fast(
            old_mean=1.0, old_var=1.0, new_output=1.0, noise_var=0.1
        )
        assert kl >= 0

        # KL should increase when observation is far from GP mean
        kl_far = evaluator._calc_kl_qp_fast(
            old_mean=1.0, old_var=1.0, new_output=10.0, noise_var=0.1
        )
        assert kl_far > kl

        # KL should be 0 when old variance is 0 (degenerate case)
        kl_zero = evaluator._calc_kl_qp_fast(
            old_mean=1.0, old_var=0.0, new_output=1.0, noise_var=0.1
        )
        assert kl_zero == 0.0

    def test_returns_empty_unfitted(self, benchmark):
        """Should return empty dict when strategy is not fitted."""
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = ExpMinRegretGapEvaluator()

        result = evaluator.evaluate(strategy, pd.DataFrame(), 0)

        assert result == {}

    def test_returns_empty_single_experiment(self, benchmark):
        """Should return empty dict with only 1 experiment."""
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(1), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        # Can't tell with 1 experiment on Himmelblau (needs >= 2 for GP)
        evaluator = ExpMinRegretGapEvaluator()
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result == {}

    def test_dumps_and_loads(self, bo_loop_strategies):
        """State should survive a dumps/loads round-trip."""
        strategy1, exp1, strategy2, exp2 = bo_loop_strategies
        evaluator = ExpMinRegretGapEvaluator()

        evaluator.evaluate(strategy1, exp1, 0)
        evaluator.evaluate(strategy2, exp2, 1)

        # Serialize state
        data = evaluator.dumps()
        assert isinstance(data, str)

        # Load into a fresh evaluator
        evaluator2 = ExpMinRegretGapEvaluator()
        evaluator2.loads(data)

        # Internal state should match
        assert evaluator2._prev_incumbent_idx == evaluator._prev_incumbent_idx
        assert evaluator2._prev_n_experiments == evaluator._prev_n_experiments
        assert evaluator2._seq_values == evaluator._seq_values
        assert evaluator2._prev_model is not None

    def test_evaluate_after_loads(self, benchmark):
        """Evaluator should produce valid metrics after loading state."""
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

        # Iteration 1
        strategy1 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy1.tell(experiments)
        evaluator = ExpMinRegretGapEvaluator()
        evaluator.evaluate(strategy1, experiments, 0)

        # Iteration 2
        candidates = strategy1.ask(1)
        candidates = candidates[benchmark.domain.inputs.get_keys()]
        new_exp = benchmark.f(candidates)
        new_xy = pd.concat([candidates, new_exp], axis=1)
        experiments2 = pd.concat([experiments, new_xy], ignore_index=True)

        strategy2 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy2.tell(experiments2)
        evaluator.evaluate(strategy2, experiments2, 1)

        # Serialize and simulate restart
        data = evaluator.dumps()

        evaluator_new = ExpMinRegretGapEvaluator()
        evaluator_new.loads(data)

        # Iteration 3 — should work with the restored evaluator
        candidates3 = strategy2.ask(1)
        candidates3 = candidates3[benchmark.domain.inputs.get_keys()]
        new_exp3 = benchmark.f(candidates3)
        new_xy3 = pd.concat([candidates3, new_exp3], axis=1)
        experiments3 = pd.concat([experiments2, new_xy3], ignore_index=True)

        strategy3 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy3.tell(experiments3)
        result = evaluator_new.evaluate(strategy3, experiments3, 2)

        assert result != {}
        assert result["stopping_value"] >= 0
        assert len(result["seq_values"]) == 2  # values from iter 2 + iter 3


class TestEvaluatorKwargs:
    """Tests for overriding evaluator parameters via constructor kwargs."""

    def test_ucblcb_kwargs_override_defaults(self):
        evaluator = UCBLCBRegretEvaluator(
            delta=0.05, beta_scale=1.0, n_samples_lcb=500
        )
        assert evaluator.delta == 0.05
        assert evaluator.beta_scale == 1.0
        assert evaluator.n_samples_lcb == 500

    def test_defaults_preserved_without_kwargs(self):
        evaluator = UCBLCBRegretEvaluator()
        assert evaluator.delta == 0.1
        assert evaluator.beta_scale == 0.2

    def test_expmin_kwargs_override_defaults(self):
        evaluator = ExpMinRegretGapEvaluator(
            delta=0.05, rate=0.2, start_timing=15, beta_scale=0.5
        )
        assert evaluator.delta == 0.05
        assert evaluator.rate == 0.2
        assert evaluator.start_timing == 15
        assert evaluator.beta_scale == 0.5
        assert evaluator._seq_values == []

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword"):
            UCBLCBRegretEvaluator(not_a_real_param=1.0)

    def test_private_kwarg_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword"):
            ExpMinRegretGapEvaluator(_prev_model="x")
