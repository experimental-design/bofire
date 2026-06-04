"""Tests for termination evaluators."""

import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.strategies.api import RandomStrategy, SoboStrategy
from bofire.strategies.stepwise.termination.exp_min_regret_gap import (
    ExpMinRegretGapEvaluator,
)
from bofire.strategies.stepwise.termination.log_eipc import LogEIPCEvaluator
from bofire.strategies.stepwise.termination.probabilistic_regret_bound import (
    ProbabilisticRegretBoundEvaluator,
    _minimize_sample_paths,
    _run_prb_level_test,
)
from bofire.strategies.stepwise.termination.ucb_lcb import UCBLCBRegretEvaluator
from bofire.strategies.stepwise.termination.utils import clopper_pearson_ci


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

        # Seed the initial design and the acquisition optimizer so the whole
        # run is deterministic — otherwise the "generally decreases" check is
        # flaky across runs.
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain, seed=42)
        )
        experiments = benchmark.f(random_strategy.ask(20), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain, seed=42)
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
            [experiments1, new_xy],
            ignore_index=True,
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
                [experiments, new_xy],
                ignore_index=True,
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
                [experiments, new_xy],
                ignore_index=True,
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


class TestLogEIPCEvaluator:
    """Unit tests for the LogEIPCEvaluator (Xie et al., 2025)."""

    def test_evaluate_returns_expected_keys(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert set(result.keys()) == {
            "max_log_eipc",
            "best_f",
            "cost_estimate",
            "lambda_cost",
        }

    def test_max_log_eipc_is_float(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert isinstance(result["max_log_eipc"], float)

    def test_best_f_equals_min_observed(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator()
        output_key = strategy.domain.outputs.get_keys()[0]

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["best_f"] == pytest.approx(experiments[output_key].min())

    def test_returns_empty_when_not_fitted(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = LogEIPCEvaluator()

        result = evaluator.evaluate(strategy, pd.DataFrame(), 0)

        assert result == {}

    def test_returns_empty_with_few_experiments(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator()

        result = evaluator.evaluate(strategy, experiments.iloc[:1], 0)

        assert result == {}

    def test_cost_column_used_when_present(self, trained_strategy):
        strategy, experiments = trained_strategy
        experiments = experiments.copy()
        experiments["cost"] = 2.5

        evaluator = LogEIPCEvaluator(cost_column="cost")
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["cost_estimate"] == pytest.approx(2.5)

    def test_cost_value_fallback_when_no_column(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator(cost_value=3.0)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["cost_estimate"] == pytest.approx(3.0)

    def test_higher_lambda_cost_decreases_max_log_eipc(self, trained_strategy):
        """Higher lambda_cost shifts the threshold up, lowering max_log_eipc."""
        strategy, experiments = trained_strategy

        result_low = LogEIPCEvaluator(lambda_cost=0.01).evaluate(
            strategy, experiments, 0
        )
        result_high = LogEIPCEvaluator(lambda_cost=100.0).evaluate(
            strategy, experiments, 0
        )

        assert result_high["max_log_eipc"] < result_low["max_log_eipc"]

    def test_returns_empty_with_zero_cost(self, trained_strategy):
        """Zero cost_value should return empty dict (undefined log(0))."""
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator(cost_value=0.0)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result == {}

    def test_cost_callable_used_when_provided(self, trained_strategy):
        """cost_callable should override cost_value for per-point evaluation."""
        strategy, experiments = trained_strategy
        # Callable returns constant 2.5 — should give same result as cost_value=2.5
        ev_callable = LogEIPCEvaluator(
            cost_callable=lambda X: X.new_full((X.shape[0],), 2.5)
        )
        ev_scalar = LogEIPCEvaluator(cost_value=2.5)

        r_callable = ev_callable.evaluate(strategy, experiments, 0)
        r_scalar = ev_scalar.evaluate(strategy, experiments, 0)

        assert r_callable["max_log_eipc"] == pytest.approx(
            r_scalar["max_log_eipc"], abs=0.5
        )

    def test_search_method_optimize_returns_valid_result(self, trained_strategy):
        """search_method='optimize' should return a valid max_log_eipc."""
        strategy, experiments = trained_strategy
        evaluator = LogEIPCEvaluator(search_method="optimize")

        result = evaluator.evaluate(strategy, experiments, 0)

        assert isinstance(result["max_log_eipc"], float)

    def test_search_method_sample_and_optimize_agree(self, trained_strategy):
        """'sample' and 'optimize' should produce results with the same sign."""
        strategy, experiments = trained_strategy

        r_sample = LogEIPCEvaluator(
            lambda_cost=1.0, cost_value=1.0, search_method="sample"
        ).evaluate(strategy, experiments, 0)
        r_opt = LogEIPCEvaluator(
            lambda_cost=1.0, cost_value=1.0, search_method="optimize"
        ).evaluate(strategy, experiments, 0)

        # Both should agree on stop vs continue
        assert (r_sample["max_log_eipc"] > 0) == (r_opt["max_log_eipc"] > 0)

    def test_cost_model_gp_returns_valid_result(self, trained_strategy):
        """cost_model='gp' should fit a cost GP and return a valid result."""
        strategy, experiments = trained_strategy
        experiments = experiments.copy()
        experiments["cost"] = np.random.uniform(1.0, 3.0, len(experiments))

        evaluator = LogEIPCEvaluator(cost_column="cost", cost_model="gp")
        result = evaluator.evaluate(strategy, experiments, 0)

        assert isinstance(result["max_log_eipc"], float)

    def test_cost_model_gp_doesnt_mutate_cost_callable(self, trained_strategy):
        """cost_callable should be None after evaluate() even with cost_model='gp'."""
        strategy, experiments = trained_strategy
        experiments = experiments.copy()
        experiments["cost"] = 2.0

        evaluator = LogEIPCEvaluator(cost_column="cost", cost_model="gp")
        assert evaluator.cost_callable is None
        evaluator.evaluate(strategy, experiments, 0)
        assert evaluator.cost_callable is None  # restored after call

    def test_cost_model_gp_vs_mean_differ(self, trained_strategy):
        """GP cost model should give a different result from scalar mean."""
        strategy, experiments = trained_strategy
        experiments = experiments.copy()
        # Costs increase with x_1 — spatial variation the GP can learn
        experiments["cost"] = 1.0 + experiments["x_1"].abs()

        r_mean = LogEIPCEvaluator(cost_column="cost", cost_model="mean").evaluate(
            strategy, experiments, 0
        )
        r_gp = LogEIPCEvaluator(cost_column="cost", cost_model="gp").evaluate(
            strategy, experiments, 0
        )

        # GP-based costs are per-point so max_log_eipc will generally differ
        assert r_mean["max_log_eipc"] != pytest.approx(r_gp["max_log_eipc"])


class TestEvaluatorKwargs:
    """Tests for overriding evaluator parameters via constructor kwargs."""

    def test_ucblcb_kwargs_override_defaults(self):
        evaluator = UCBLCBRegretEvaluator(delta=0.05, beta_scale=1.0, n_samples_lcb=500)
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


# ---------------------------------------------------------------------------
# PRB helper unit tests
# ---------------------------------------------------------------------------


class TestClopperPearsonCI:
    """Unit tests for the clopper_pearson_ci helper."""

    def test_zero_successes_gives_lower_zero(self):
        lower, upper = clopper_pearson_ci(k=0, n=20, risk=0.05)
        assert lower == 0.0
        assert 0.0 < upper < 1.0

    def test_all_successes_gives_upper_one(self):
        lower, upper = clopper_pearson_ci(k=20, n=20, risk=0.05)
        assert lower > 0.0
        assert upper == 1.0

    def test_bounds_contain_empirical_rate(self):
        """For k=5 out of n=20, both bounds should be > 0 and < 1."""
        lower, upper = clopper_pearson_ci(k=5, n=20, risk=0.05)
        assert 0.0 < lower < upper < 1.0

    def test_lower_leq_upper(self):
        for k in [0, 5, 10, 15, 20]:
            lower, upper = clopper_pearson_ci(k=k, n=20, risk=0.05)
            assert lower <= upper

    def test_interval_shrinks_with_more_samples(self):
        """Larger n → narrower CI for the same proportion."""
        _, upper_small = clopper_pearson_ci(k=5, n=10, risk=0.05)
        lower_small, _ = clopper_pearson_ci(k=5, n=10, risk=0.05)
        _, upper_large = clopper_pearson_ci(k=50, n=100, risk=0.05)
        lower_large, _ = clopper_pearson_ci(k=50, n=100, risk=0.05)
        assert (upper_large - lower_large) < (upper_small - lower_small)

    def test_tighter_risk_gives_wider_interval(self):
        """Smaller risk (higher confidence) → wider CI."""
        lower_wide, upper_wide = clopper_pearson_ci(k=5, n=20, risk=0.01)
        lower_narrow, upper_narrow = clopper_pearson_ci(k=5, n=20, risk=0.20)
        assert (upper_wide - lower_wide) > (upper_narrow - lower_narrow)


class TestMinimizeSamplePaths:
    """Unit tests for _minimize_sample_paths."""

    def test_returns_correct_shape(self, trained_strategy):
        """Should return an array of length n_samples."""
        import torch
        from botorch.sampling.pathwise import draw_matheron_paths

        from bofire.utils.torch_tools import tkwargs

        strategy, experiments = trained_strategy
        model = strategy.model
        bounds = strategy.domain.inputs.get_bounds(
            specs=strategy.input_preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        n_samples = 4
        paths = draw_matheron_paths(model, sample_shape=torch.Size([n_samples]))
        minima = _minimize_sample_paths(paths, lower, upper, n_random=64, n_starts=2)

        assert minima.shape == (n_samples,)

    def test_minima_leq_path_at_evaluated_points(self, trained_strategy):
        """The minimum must be ≤ the path value at each evaluated point."""
        import torch
        from botorch.sampling.pathwise import draw_matheron_paths

        from bofire.utils.torch_tools import tkwargs

        strategy, experiments = trained_strategy
        model = strategy.model
        input_keys = strategy.domain.inputs.get_keys()
        transformed = strategy.domain.inputs.transform(
            experiments[input_keys], strategy.input_preprocessing_specs
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        bounds = strategy.domain.inputs.get_bounds(
            specs=strategy.input_preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        n_samples = 3
        paths = draw_matheron_paths(model, sample_shape=torch.Size([n_samples]))
        minima = _minimize_sample_paths(paths, lower, upper, n_random=64, n_starts=2)

        with torch.no_grad():
            vals = paths(X).numpy()  # [n_samples, n_points]

        for i in range(n_samples):
            # minimum found by optimiser must not exceed the minimum random sample
            assert minima[i] <= vals[i].min() + 1e-6


class TestRunPRBLevelTest:
    """Unit tests for _run_prb_level_test."""

    def test_all_zeros_converges_below(self):
        """When P(indicator=1)=0, all CI uppers should drop below level."""
        n_test = 1
        level = 0.1  # P(regret > ε) must be ≤ 0.1 to stop

        def sampler_fn(n_batch):
            return np.zeros((n_test, n_batch), dtype=np.int64)

        estimates, converged_below, total_n, cis = _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=level,
            delta_est=0.05,
            n_samples_max=512,
            n_test=n_test,
        )
        # P̂(indicator=1) = 0/total_n = 0 < level → converged_below should be True
        assert converged_below[0]
        assert float(estimates[0]) == pytest.approx(0.0)

    def test_all_ones_does_not_converge_below(self):
        """When P(indicator=1)=1 >> level, no test point satisfies the criterion."""
        n_test = 1
        level = 0.1

        def sampler_fn(n_batch):
            return np.ones((n_test, n_batch), dtype=np.int64)

        estimates, converged_below, total_n, cis = _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=level,
            delta_est=0.05,
            n_samples_max=512,
            n_test=n_test,
        )
        assert not converged_below[0]
        assert float(estimates[0]) == pytest.approx(1.0)

    def test_respects_n_samples_max(self):
        """Should never draw more than n_samples_max total indicators."""
        n_test = 1
        calls = []

        def sampler_fn(n_batch):
            calls.append(n_batch)
            return np.zeros((n_test, n_batch), dtype=np.int64)

        _, _, total_n, _ = _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=0.5,  # hard to satisfy quickly
            delta_est=0.05,
            n_samples_max=128,
            n_test=n_test,
        )
        assert total_n <= 128

    def test_multiple_test_points(self):
        """With n_test > 1, best test point should dominate."""
        n_test = 3
        level = 0.1

        def sampler_fn(n_batch):
            # Point 0: always 0 (criterion met); points 1, 2: always 1
            indicators = np.ones((n_test, n_batch), dtype=np.int64)
            indicators[0] = 0
            return indicators

        estimates, converged_below, total_n, cis = _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=level,
            delta_est=0.05,
            n_samples_max=512,
            n_test=n_test,
        )
        # Point 0 has 0 failures → converged below
        assert converged_below[0]
        # Points 1 and 2 have all failures → not converged below
        assert not converged_below[1]
        assert not converged_below[2]

    def test_ci_contains_estimate(self):
        """The CI should contain the empirical estimate."""
        n_test = 1

        def sampler_fn(n_batch):
            rng = np.random.default_rng(0)
            return rng.integers(0, 2, size=(n_test, n_batch))

        estimates, _, _, cis = _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=0.5,
            delta_est=0.05,
            n_samples_max=256,
            n_test=n_test,
        )
        assert cis[0, 0] <= estimates[0] <= cis[0, 1]


class TestProbabilisticRegretBoundEvaluator:
    """Integration tests for ProbabilisticRegretBoundEvaluator."""

    # Use small resource budgets to keep tests fast.
    _fast_kwargs = {
        "n_samples_max": 32,
        "n_random": 64,
        "n_starts": 2,
        "initial_batch": 16,
    }

    def test_returns_expected_keys(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        expected = {
            "prob_regret_ok",
            "epsilon",
            "delta_mod",
            "criterion_satisfied",
            "n_samples_used",
            "ci_lower",
            "ci_upper",
            "converged",
        }
        assert expected == set(result.keys())

    def test_prob_regret_ok_in_unit_interval(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert 0.0 <= result["prob_regret_ok"] <= 1.0

    def test_ci_valid(self, trained_strategy):
        """CI should satisfy 0 ≤ lower ≤ estimate ≤ upper ≤ 1."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert 0.0 <= result["ci_lower"]
        assert result["ci_lower"] <= result["prob_regret_ok"] + 1e-9
        assert result["prob_regret_ok"] <= result["ci_upper"] + 1e-9
        assert result["ci_upper"] <= 1.0

    def test_criterion_satisfied_is_bool(self, trained_strategy):
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert isinstance(result["criterion_satisfied"], bool)

    def test_n_samples_used_leq_max(self, trained_strategy):
        """Should never exceed n_samples_max total path samples."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["n_samples_used"] <= self._fast_kwargs["n_samples_max"]

    def test_epsilon_computed_from_relative_when_not_set(self, trained_strategy):
        """When epsilon is None, ε = epsilon_relative * (y_max - y_min)."""
        strategy, experiments = trained_strategy
        output_key = strategy.domain.outputs.get_keys()[0]
        y_range = float(experiments[output_key].max() - experiments[output_key].min())

        evaluator = ProbabilisticRegretBoundEvaluator(
            epsilon=None, epsilon_relative=0.05, **self._fast_kwargs
        )
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["epsilon"] == pytest.approx(0.05 * y_range, rel=1e-6)

    def test_epsilon_override_used_when_set(self, trained_strategy):
        """When epsilon is set, it must be used directly."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(epsilon=2.5, **self._fast_kwargs)
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["epsilon"] == pytest.approx(2.5)

    def test_very_large_epsilon_gives_high_prob(self, trained_strategy):
        """ε much larger than any plausible regret → P(regret ≤ ε) should be high."""
        strategy, experiments = trained_strategy
        output_key = strategy.domain.outputs.get_keys()[0]
        y_range = float(experiments[output_key].max() - experiments[output_key].min())

        evaluator = ProbabilisticRegretBoundEvaluator(
            epsilon=100.0 * y_range,  # absurdly large ε
            **self._fast_kwargs,
        )
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["prob_regret_ok"] > 0.8

    def test_very_small_epsilon_gives_low_prob(self, trained_strategy):
        """ε = 0 → the path minimum always equals the optimum → P(regret ≤ 0) = 0."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(
            epsilon=0.0,
            enforce_convergence=False,
            **self._fast_kwargs,
        )
        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["prob_regret_ok"] < 0.3

    def test_returns_empty_when_not_fitted(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, pd.DataFrame(), 0)

        assert result == {}

    def test_returns_empty_with_single_experiment(self, benchmark):
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(1), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        evaluator = ProbabilisticRegretBoundEvaluator(**self._fast_kwargs)

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result == {}

    def test_n_test_points_greater_than_one(self, trained_strategy):
        """n_test_points > 1 should still produce a valid result."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(
            n_test_points=3, **self._fast_kwargs
        )
        result = evaluator.evaluate(strategy, experiments, 0)

        assert 0.0 <= result["prob_regret_ok"] <= 1.0

    def test_enforce_convergence_false_uses_raw_estimate(self, trained_strategy):
        """With enforce_convergence=False, criterion_satisfied depends on raw estimate."""
        strategy, experiments = trained_strategy
        evaluator = ProbabilisticRegretBoundEvaluator(
            epsilon=1000.0,  # huge ε: prob_regret_ok should be ~1.0
            delta_mod=0.5,  # stop when P(regret > ε) ≤ 0.5 (very loose)
            enforce_convergence=False,
            **self._fast_kwargs,
        )
        result = evaluator.evaluate(strategy, experiments, 0)

        # With ε=1000 (all regrets < ε), P(regret > ε) ≈ 0 ≤ delta_mod=0.5 → True
        assert result["criterion_satisfied"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Conformance tests — verify implementation matches Wilson (2024) / trieste_stopping
# ──────────────────────────────────────────────────────────────────────────────


class TestClopperPearsonExactFormula:
    """Verify clopper_pearson_ci matches the analytical beta-distribution formula
    from Clopper & Pearson (1934).  The reference implementation in trieste_stopping
    uses the same formula; any deviation here means the CI bounds diverge."""

    def test_matches_scipy_beta_ppf(self):
        """Lower = B(α/2; k, n−k+1) and upper = B(1−α/2; k+1, n−k)."""
        from scipy.stats import beta as scipy_beta

        k, n, risk = 7, 30, 0.05
        lower, upper = clopper_pearson_ci(k, n, risk)

        expected_lower = float(scipy_beta.ppf(risk / 2.0, k, n - k + 1))
        expected_upper = float(scipy_beta.ppf(1.0 - risk / 2.0, k + 1, n - k))

        assert abs(lower - expected_lower) < 1e-10, (
            f"lower={lower:.10f} != expected={expected_lower:.10f}"
        )
        assert abs(upper - expected_upper) < 1e-10, (
            f"upper={upper:.10f} != expected={expected_upper:.10f}"
        )

    def test_boundary_k0_lower_exactly_zero(self):
        """k=0 → lower must be *exactly* 0.0, not just near-zero."""
        lower, _ = clopper_pearson_ci(k=0, n=50, risk=0.05)
        assert lower == 0.0

    def test_boundary_kn_upper_exactly_one(self):
        """k=n → upper must be *exactly* 1.0, not just near-one."""
        _, upper = clopper_pearson_ci(k=50, n=50, risk=0.05)
        assert upper == 1.0

    def test_symmetric_around_half(self):
        """For k=n/2 with symmetric risk, (lower + upper) / 2 ≈ 0.5."""
        lower, upper = clopper_pearson_ci(k=50, n=100, risk=0.05)
        assert abs((lower + upper) / 2.0 - 0.5) < 0.01


class TestRunPRBLevelTestScheduleAndGuarantee:
    """Verify Algorithm 2 schedule parameters from Wilson (2024) Section 3.2.

    Paper: d_j = ((α−1)/α) · δ_est · j^(−α), n_j = ⌈β^(j−1) · N⌉, α=1.1, β=1.5.
    Reference code: trieste_stopping/stopping/utils/convergence.py."""

    def test_batch_growth_schedule(self):
        """Batches are increments of cumulative targets n_j = int(N·β^(j−1)).

        Matching trieste_stopping: the schedule value is the *cumulative* target,
        so the batch drawn at step j is n_j − n_{j−1}, not n_j itself.
        Cumulative totals: 16, 24, 36, 54, …  Batches: 16, 8, 12, 18, …
        """
        batches_seen = []

        def sampler_fn(n_batch):
            batches_seen.append(n_batch)
            # p≈0.5 so convergence takes several steps
            rng = np.random.default_rng(0)
            return rng.integers(0, 2, size=(1, n_batch))

        initial_batch, batch_growth, n_samples_max = 16, 1.5, 300

        _run_prb_level_test(
            sampler_fn=sampler_fn,
            level=0.5,
            delta_est=0.05,
            n_samples_max=n_samples_max,
            n_test=1,
            initial_batch=initial_batch,
            batch_growth=batch_growth,
        )

        cumul = 0
        for j, actual in enumerate(batches_seen):
            n_target = int(initial_batch * batch_growth**j)
            expected = min(n_target - cumul, n_samples_max - cumul)
            assert actual == expected, (
                f"Step {j + 1}: expected increment={expected} "
                f"(cumul target {n_target}), got batch={actual}"
            )
            cumul += actual

    def test_per_step_risk_schedule_sums_to_leq_delta_est(self):
        """Σ_j d_j = Σ_j ((α−1)/α) · δ_est · j^(−α) ≤ δ_est for α=1.1.

        This is the key budget constraint from Wilson (2024) Section 3.2 that
        guarantees the total false-positive risk is bounded."""
        delta_est = 0.1
        alpha = 1.1

        partial_sum = sum(
            ((alpha - 1) / alpha) * delta_est * (step**-alpha)
            for step in range(1, 10_000)
        )
        assert partial_sum <= delta_est + 1e-6, (
            f"Risk schedule partial sum {partial_sum:.6f} exceeds δ_est={delta_est}"
        )

    def test_type1_error_rate_bounded_by_delta_est(self):
        """When true p >> level, the false-positive rate must be ≤ δ_est.

        False positive = CP test incorrectly concludes P(indicator=1) ≤ level
        when the true probability is much higher.  Wilson (2024) Proposition 2
        guarantees this rate is ≤ δ_est."""
        true_p = 0.9  # well above level
        level = 0.1
        delta_est = 0.05
        n_trials = 200
        rng = np.random.default_rng(20240101)

        false_positives = sum(
            _run_prb_level_test(
                sampler_fn=lambda n, _r=rng: (_r.random((1, n)) < true_p).astype(
                    np.int64
                ),
                level=level,
                delta_est=delta_est,
                n_samples_max=512,
                n_test=1,
            )[1][0]  # converged_below[0]
            for _ in range(n_trials)
        )

        # The theoretical bound is delta_est; allow 3× slack for finite-sample
        # randomness — a genuinely buggy implementation would far exceed this.
        observed_rate = false_positives / n_trials
        assert observed_rate <= 3 * delta_est, (
            f"Type-I error {false_positives}/{n_trials}={observed_rate:.3f} "
            f"exceeds 3·δ_est={3 * delta_est:.3f}"
        )


class TestPRBKnownAnswerEstimation:
    """Known-answer Ψ tests: compare _evaluate_core against analytical expectations.

    Setup: f(x) = cos(2πx) on [0,1] with near-noiseless observations.
      - True minimum  at x=0.5  (f=−1): regret≈0  → Ψ(x;ε=0.1) ≈ 1
      - True maximum  at x=0.0  (f=+1): regret≈2  → Ψ(x;ε=0.1) ≈ 0

    This catches bugs in: regret sign, path minimisation, ε threshold, and the
    conversion from P(regret>ε) to prob_regret_ok.
    """

    @pytest.fixture
    def cosine_gp(self):
        """Near-noiseless GP fitted to cos(2πx) at 9 equally-spaced points."""
        import torch
        from botorch.models import SingleTaskGP
        from gpytorch.kernels import MaternKernel, ScaleKernel

        NOISE_VAR = 1e-4
        X_obs = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
        Y_obs = torch.cos(torch.tensor(2 * np.pi) * X_obs)

        covar = ScaleKernel(MaternKernel(nu=2.5))
        gp = SingleTaskGP(X_obs, Y_obs, covar_module=covar)
        with torch.no_grad():
            gp.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
            gp.covar_module.base_kernel.lengthscale = torch.tensor(
                0.25, dtype=torch.float64
            )
            gp.likelihood.noise = torch.tensor(NOISE_VAR, dtype=torch.float64)
        gp.eval()
        return gp

    def _make_evaluator(self, epsilon):
        return ProbabilisticRegretBoundEvaluator(
            epsilon=epsilon,
            enforce_convergence=False,
            n_samples_max=1500,
            n_random=128,
            n_starts=4,
        )

    def test_psi_near_one_at_true_minimum(self, cosine_gp):
        """At x=0.5 (f=−1, true minimum): regret≈0 → Ψ should be close to 1."""
        import torch

        lower = torch.zeros(1, dtype=torch.float64)
        upper = torch.ones(1, dtype=torch.float64)
        X_min = torch.tensor([[0.5]], dtype=torch.float64)

        result = self._make_evaluator(0.1)._evaluate_core(
            cosine_gp, X_min, lower, upper, epsilon=0.1
        )

        assert result["prob_regret_ok"] > 0.85, (
            f"Ψ at true minimum should be ≈1 (got {result['prob_regret_ok']:.3f}). "
            "Likely cause: regret formula inverted or path minimisation broken."
        )

    def test_psi_near_zero_at_true_maximum(self, cosine_gp):
        """At x=0.0 (f=+1, true maximum): regret≈2 >> ε=0.1 → Ψ should be ≈0."""
        import torch

        lower = torch.zeros(1, dtype=torch.float64)
        upper = torch.ones(1, dtype=torch.float64)
        X_max = torch.tensor([[0.0]], dtype=torch.float64)

        result = self._make_evaluator(0.1)._evaluate_core(
            cosine_gp, X_max, lower, upper, epsilon=0.1
        )

        assert result["prob_regret_ok"] < 0.10, (
            f"Ψ at true maximum should be ≈0 (got {result['prob_regret_ok']:.3f}). "
            "Likely cause: regret sign wrong (maximising instead of minimising)."
        )

    def test_psi_monotone_increasing_in_epsilon(self, cosine_gp):
        """Ψ(x; ε) must be non-decreasing in ε for any fixed x.

        Monotonicity is a basic correctness requirement: a larger tolerance
        can only make more paths satisfy regret ≤ ε."""
        import torch

        lower = torch.zeros(1, dtype=torch.float64)
        upper = torch.ones(1, dtype=torch.float64)
        # x=0.25: intermediate regret, so Ψ should transition from low to high
        X_test = torch.tensor([[0.25]], dtype=torch.float64)

        psi_values = []
        for eps in [0.05, 0.5, 2.0]:
            ev = ProbabilisticRegretBoundEvaluator(
                epsilon=eps,
                enforce_convergence=False,
                n_samples_max=500,
                n_random=64,
                n_starts=2,
            )
            result = ev._evaluate_core(cosine_gp, X_test, lower, upper, epsilon=eps)
            psi_values.append(result["prob_regret_ok"])

        assert psi_values[0] <= psi_values[1] + 0.05, (
            f"Ψ not monotone: ε=0.05→{psi_values[0]:.3f}, ε=0.5→{psi_values[1]:.3f}"
        )
        assert psi_values[1] <= psi_values[2] + 0.05, (
            f"Ψ not monotone: ε=0.5→{psi_values[1]:.3f}, ε=2.0→{psi_values[2]:.3f}"
        )


def _fit_strategy(objective, X, y, seed=0):
    """Fit a SoboStrategy on (X, y) with a single output using ``objective``."""
    import torch

    from bofire.data_models.domain.api import Domain, Inputs, Outputs
    from bofire.data_models.features.api import ContinuousInput, ContinuousOutput

    torch.manual_seed(seed)
    d = X.shape[1]
    domain = Domain(
        inputs=Inputs(
            features=[ContinuousInput(key=f"x{i}", bounds=(0.0, 1.0)) for i in range(d)]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y", objective=objective)]),
    )
    exp = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    exp["y"] = y
    exp["valid_y"] = 1
    strategy = SoboStrategy(data_model=SoboStrategyDataModel(domain=domain))
    strategy.tell(exp)
    return strategy, exp


class TestObjectiveDirection:
    """Termination evaluators support both minimisation and maximisation.

    The core correctness check is **negation invariance**: minimising ``y`` and
    maximising ``-y`` are the same optimisation problem, and BoFire's GP
    posterior mean is linear in the (standardised) targets, so a correct
    direction-aware evaluator must produce identical metrics for
    ``(MinimizeObjective, y)`` and ``(MaximizeObjective, -y)``.
    """

    def _xy(self):
        rng = np.random.default_rng(1)
        X = rng.random((12, 2))
        # A smooth, non-trivial response so the regret bounds are well-defined.
        y = (X[:, 0] - 0.4) ** 2 + 0.5 * (X[:, 1] - 0.7) ** 2
        return X, y

    def _pair(self):
        from bofire.data_models.objectives.api import (
            MaximizeObjective,
            MinimizeObjective,
        )

        X, y = self._xy()
        strat_min, exp_min = _fit_strategy(MinimizeObjective(), X, y)
        strat_max, exp_max = _fit_strategy(MaximizeObjective(), X, -y)
        return (strat_min, exp_min), (strat_max, exp_max)

    def test_ucblcb_negation_invariance(self):
        import torch

        (strat_min, exp_min), (strat_max, exp_max) = self._pair()
        ev = UCBLCBRegretEvaluator()
        torch.manual_seed(7)
        m_min = ev.evaluate(strat_min, exp_min, 0)
        torch.manual_seed(7)
        m_max = ev.evaluate(strat_max, exp_max, 0)

        assert m_min and m_max  # neither rejected
        assert m_min["regret_bound"] == pytest.approx(
            m_max["regret_bound"], rel=0.02, abs=1e-3
        )

    def test_logeipc_negation_invariance(self):
        import torch

        (strat_min, exp_min), (strat_max, exp_max) = self._pair()
        ev = LogEIPCEvaluator(n_samples=2000)
        torch.manual_seed(7)
        m_min = ev.evaluate(strat_min, exp_min, 0)
        torch.manual_seed(7)
        m_max = ev.evaluate(strat_max, exp_max, 0)

        assert m_min and m_max
        assert m_min["max_log_eipc"] == pytest.approx(
            m_max["max_log_eipc"], rel=0.02, abs=1e-3
        )
        # best_f flips sign between the two framings.
        assert m_min["best_f"] == pytest.approx(-m_max["best_f"], rel=1e-6, abs=1e-9)

    def test_expminregretgap_negation_invariance(self):
        """Stateful: first call seeds state, second (after a new point) scores."""
        import torch

        from bofire.data_models.objectives.api import (
            MaximizeObjective,
            MinimizeObjective,
        )

        X, y = self._xy()
        x_new = np.array([[0.42, 0.68]])
        y_new = float((x_new[0, 0] - 0.4) ** 2 + 0.5 * (x_new[0, 1] - 0.7) ** 2)

        def run(objective, sign):
            strat, exp = _fit_strategy(objective, X, sign * y)
            ev = ExpMinRegretGapEvaluator()
            torch.manual_seed(7)
            assert ev.evaluate(strat, exp, 0) == {}  # first call seeds state
            exp2 = pd.concat(
                [
                    exp,
                    pd.DataFrame(
                        {
                            "x0": x_new[:, 0],
                            "x1": x_new[:, 1],
                            "y": [sign * y_new],
                            "valid_y": [1],
                        }
                    ),
                ],
                ignore_index=True,
            )
            strat.tell(exp2)
            torch.manual_seed(7)
            return ev.evaluate(strat, exp2, 1)

        m_min = run(MinimizeObjective(), 1.0)
        m_max = run(MaximizeObjective(), -1.0)
        assert m_min and m_max
        assert m_min["stopping_value"] == pytest.approx(
            m_max["stopping_value"], rel=0.05, abs=1e-3
        )

    def test_prb_maximization_known_answer(self):
        """PRB on a maximisation problem: f(x)=cos(2πx), global max at x=0.

        At the true maximum (x=0) regret≈0 → Ψ≈1; at the true minimum
        (x=0.5, the worst point for maximisation) regret≈2 → Ψ≈0.
        """
        import torch
        from botorch.models import SingleTaskGP
        from gpytorch.kernels import MaternKernel, ScaleKernel

        X_obs = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
        Y_obs = torch.cos(torch.tensor(2 * np.pi) * X_obs)
        gp = SingleTaskGP(X_obs, Y_obs, covar_module=ScaleKernel(MaternKernel(nu=2.5)))
        with torch.no_grad():
            gp.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
            gp.covar_module.base_kernel.lengthscale = torch.tensor(
                0.25, dtype=torch.float64
            )
            gp.likelihood.noise = torch.tensor(1e-4, dtype=torch.float64)
        gp.eval()

        lower = torch.zeros(1, dtype=torch.float64)
        upper = torch.ones(1, dtype=torch.float64)
        ev = ProbabilisticRegretBoundEvaluator(
            epsilon=0.1,
            enforce_convergence=False,
            n_samples_max=1500,
            n_random=128,
            n_starts=4,
        )

        at_max = ev._evaluate_core(
            gp, torch.tensor([[0.0]], dtype=torch.float64), lower, upper, 0.1, sign=-1.0
        )
        at_min = ev._evaluate_core(
            gp, torch.tensor([[0.5]], dtype=torch.float64), lower, upper, 0.1, sign=-1.0
        )
        assert at_max["prob_regret_ok"] > 0.85, (
            f"Ψ at the true maximum should be ≈1, got {at_max['prob_regret_ok']:.3f}"
        )
        assert at_min["prob_regret_ok"] < 0.10, (
            f"Ψ at the true minimum should be ≈0, got {at_min['prob_regret_ok']:.3f}"
        )

    def test_unsupported_objective_rejected(self):
        """Objectives that are neither Minimize nor Maximize → empty metrics."""
        from bofire.data_models.objectives.api import CloseToTargetObjective

        X, y = self._xy()
        strat, exp = _fit_strategy(
            CloseToTargetObjective(target_value=1.0, exponent=2), X, y
        )
        evaluators = [
            UCBLCBRegretEvaluator(),
            ExpMinRegretGapEvaluator(),
            LogEIPCEvaluator(),
            ProbabilisticRegretBoundEvaluator(
                n_samples_max=32, n_random=64, n_starts=2
            ),
        ]
        for ev in evaluators:
            assert ev.evaluate(strat, exp, 0) == {}, (
                f"{type(ev).__name__} should reject a CloseToTargetObjective"
            )
