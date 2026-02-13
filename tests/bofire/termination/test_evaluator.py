"""Tests for termination evaluators."""

import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.api import qUCB
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.data_models.termination.api import UCBLCBRegretTermination
from bofire.runners.api import RunResult, run
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

    def test_beta_from_qucb(self, benchmark):
        """When strategy uses qUCB, the evaluator should use its beta."""
        custom_beta = 0.5
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(
                domain=benchmark.domain,
                acquisition_function=qUCB(beta=custom_beta),
            )
        )
        strategy.tell(experiments)
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["beta"] == custom_beta

    def test_fallback_beta_used_when_no_qucb(self, trained_strategy):
        """When strategy doesn't use qUCB, fallback beta (0.2) is used."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["beta"] == evaluator.fallback_beta

    def test_noise_variance_estimated(self, trained_strategy):
        """Noise variance should be estimated from the GP likelihood."""
        strategy, experiments = trained_strategy
        evaluator = UCBLCBRegretEvaluator()

        result = evaluator.evaluate(strategy, experiments, 0)

        assert result["estimated_noise_variance"] > 0


class TestMakarovaMethodIntegration:
    """Integration tests for the full Makarova termination method.

    These tests verify that the evaluator + termination condition work
    together correctly through the BO loop.
    """

    def test_regret_bound_generally_decreases(self, benchmark):
        """Over many BO iterations, the regret bound should generally decrease.

        After fitting with more data, the GP becomes more certain and the
        gap between UCB and LCB shrinks. We check that the final regret
        bound is smaller than the initial one.
        """
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)
        evaluator = UCBLCBRegretEvaluator()

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

        # The final regret bound should be smaller than the initial one
        assert regret_bounds[-1] < regret_bounds[0]

    def test_early_termination_via_run(self, benchmark):
        """Run with a termination condition should stop before max iterations."""

        def metric(domain, experiments):
            return float(experiments["y"].min())

        # Use a very generous threshold so termination happens quickly.
        # The noise variance estimated from GP is ~1e-2 to 1e-4,
        # so threshold = 1e6 * noise_var gives epsilon ~ 1e2 to 1e4.
        termination = UCBLCBRegretTermination(
            threshold_factor=1e6,
            min_iterations=2,
        )

        def sample(domain):
            sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
            return sampler.ask(10)

        results = run(
            benchmark,
            strategy_factory=lambda domain: SoboStrategy(
                data_model=SoboStrategyDataModel(domain=domain)
            ),
            n_iterations=50,
            metric=metric,
            initial_sampler=sample,
            n_runs=1,
            n_procs=1,
            termination_condition=termination,
        )

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, RunResult)
        # With a very generous threshold, should terminate early
        assert result.terminated_early is True
        assert result.final_iteration < 49

    def test_run_without_termination_completes_all_iterations(self, benchmark):
        """Without a termination condition, all iterations should run."""

        def metric(domain, experiments):
            return float(experiments["y"].min())

        n_iterations = 3

        def sample(domain):
            sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
            return sampler.ask(10)

        results = run(
            benchmark,
            strategy_factory=lambda domain: SoboStrategy(
                data_model=SoboStrategyDataModel(domain=domain)
            ),
            n_iterations=n_iterations,
            metric=metric,
            initial_sampler=sample,
            n_runs=1,
            n_procs=1,
        )

        result = results[0]
        assert result.terminated_early is False
        assert result.final_iteration == n_iterations - 1

    def test_termination_metrics_recorded_in_run_result(self, benchmark):
        """RunResult should contain termination metrics when a condition is used."""

        def metric(domain, experiments):
            return float(experiments["y"].min())

        termination = UCBLCBRegretTermination(
            threshold_factor=0.001,  # Very tight: won't terminate early
            min_iterations=2,
        )

        def sample(domain):
            sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
            return sampler.ask(10)

        results = run(
            benchmark,
            strategy_factory=lambda domain: SoboStrategy(
                data_model=SoboStrategyDataModel(domain=domain)
            ),
            n_iterations=3,
            metric=metric,
            initial_sampler=sample,
            n_runs=1,
            n_procs=1,
            termination_condition=termination,
        )

        result = results[0]
        # Termination metrics should have been recorded
        assert "regret_bound" in result.termination_metrics
        assert len(result.termination_metrics["regret_bound"]) == 3
        # All regret bounds should be non-negative
        assert all(rb >= 0 for rb in result.termination_metrics["regret_bound"])
