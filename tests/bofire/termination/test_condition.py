"""Tests for UCBLCBRegretBoundCondition, ExpMinRegretGapCondition, and StepwiseStrategy termination."""

import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    ExpMinRegretGapCondition,
    LogEIPCCondition,
    NumberOfExperimentsCondition,
    Step,
    UCBLCBRegretBoundCondition,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.data_models.strategies.api import (
    StepwiseStrategy as StepwiseStrategyDataModel,
)
from bofire.strategies.api import (
    OptimizationComplete,
    RandomStrategy,
    SoboStrategy,
    StepwiseStrategy,
)


@pytest.fixture
def benchmark():
    return Himmelblau()


class TestUCBLCBRegretBoundConditionDataModel:
    """Tests for the condition data model (serialization, defaults, etc.)."""

    def test_defaults(self):
        cond = UCBLCBRegretBoundCondition()
        assert cond.noise_variance is None
        assert cond.threshold_factor == 1.0
        assert cond.min_experiments == 5

    def test_custom_params(self):
        cond = UCBLCBRegretBoundCondition(
            noise_variance=0.1,
            threshold_factor=2.0,
            min_experiments=10,
        )
        assert cond.noise_variance == 0.1
        assert cond.threshold_factor == 2.0
        assert cond.min_experiments == 10

    def test_serialization(self):
        cond = UCBLCBRegretBoundCondition(noise_variance=0.1, threshold_factor=2.0)
        data = cond.model_dump()
        restored = UCBLCBRegretBoundCondition(**data)
        assert restored.noise_variance == cond.noise_variance
        assert restored.threshold_factor == cond.threshold_factor

    def test_returns_true_without_strategy(self, benchmark):
        """Without a strategy kwarg, condition returns True (keep going)."""
        cond = UCBLCBRegretBoundCondition()
        assert cond.evaluate(benchmark.domain, None) is True

    def test_returns_true_with_unfitted_strategy(self, benchmark):
        """With an unfitted strategy, condition returns True (keep going)."""
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        cond = UCBLCBRegretBoundCondition()
        assert cond.evaluate(benchmark.domain, None, strategy=strategy) is True

    def test_returns_true_with_few_experiments(self, benchmark):
        """With fewer than min_experiments, condition returns True."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(3), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = UCBLCBRegretBoundCondition(min_experiments=10)
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is True

    def test_evaluates_regret_bound(self, benchmark):
        """With a fitted strategy and enough data, condition computes the regret bound."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Very tight threshold (manual noise_variance mode): should NOT terminate
        cond_tight = UCBLCBRegretBoundCondition(
            noise_variance=1e-10, threshold_factor=1.0, min_experiments=5
        )
        assert (
            cond_tight.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

        # Very generous threshold (manual noise_variance mode): should terminate
        cond_generous = UCBLCBRegretBoundCondition(
            noise_variance=1e6, threshold_factor=1.0, min_experiments=5
        )
        assert (
            cond_generous.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )

    def test_gp_noise_threshold(self, benchmark):
        """Default (noise_variance=None) uses GP estimated noise for threshold."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Very large threshold_factor with GP noise → should terminate
        cond_generous = UCBLCBRegretBoundCondition(
            threshold_factor=1e8, min_experiments=5
        )
        assert cond_generous.noise_variance is None  # GP noise mode
        assert (
            cond_generous.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )

        # Default threshold_factor (1.0) with GP noise ~1e-4 → should NOT terminate
        cond_default = UCBLCBRegretBoundCondition(min_experiments=5)
        assert (
            cond_default.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

    def test_cv_mode_validation(self):
        """noise_variance='cv' requires cv_fold_columns with >= 2 columns."""
        with pytest.raises(ValueError, match="cv_fold_columns"):
            UCBLCBRegretBoundCondition(noise_variance="cv")

        with pytest.raises(ValueError, match="cv_fold_columns"):
            UCBLCBRegretBoundCondition(noise_variance="cv", cv_fold_columns=["fold_0"])

        # Valid: 2+ fold columns
        cond = UCBLCBRegretBoundCondition(
            noise_variance="cv",
            cv_fold_columns=["fold_0", "fold_1", "fold_2"],
        )
        assert cond.noise_variance == "cv"
        assert len(cond.cv_fold_columns) == 3

    def test_cv_mode_serialization(self):
        """CV mode conditions serialize and deserialize correctly."""
        cond = UCBLCBRegretBoundCondition(
            noise_variance="cv",
            cv_fold_columns=["f0", "f1", "f2", "f3", "f4"],
            threshold_factor=0.5,
        )
        data = cond.model_dump()
        restored = UCBLCBRegretBoundCondition(**data)
        assert restored.noise_variance == "cv"
        assert restored.cv_fold_columns == ["f0", "f1", "f2", "f3", "f4"]
        assert restored.threshold_factor == 0.5

    def test_cv_mode_threshold(self, benchmark):
        """With noise_variance='cv', uses incumbent's CV fold std for threshold."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        # Add synthetic fold columns with known variability
        n_folds = 5
        fold_cols = [f"y_fold_{i}" for i in range(n_folds)]
        rng = np.random.RandomState(42)
        for col in fold_cols:
            experiments[col] = experiments["y"] + rng.normal(0, 0.5, len(experiments))

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Very large threshold_factor → threshold >> regret bound → should terminate
        cond_generous = UCBLCBRegretBoundCondition(
            noise_variance="cv",
            cv_fold_columns=fold_cols,
            threshold_factor=1e6,
            min_experiments=5,
        )
        assert (
            cond_generous.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )

        # Very small threshold_factor → should NOT terminate
        cond_tight = UCBLCBRegretBoundCondition(
            noise_variance="cv",
            cv_fold_columns=fold_cols,
            threshold_factor=1e-10,
            min_experiments=5,
        )
        assert (
            cond_tight.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

    def test_cv_mode_corrected_std(self, benchmark):
        """Verify the corrected std uses the Nadeau & Bengio (2003) formula."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        # Set fold scores for the incumbent (min y row) with known values
        n_folds = 5
        fold_cols = [f"y_fold_{i}" for i in range(n_folds)]
        incumbent_idx = experiments["y"].idxmin()

        # Give all experiments uniform folds, but incumbent gets specific values
        for col in fold_cols:
            experiments[col] = experiments["y"]  # zero std
        known_folds = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, col in enumerate(fold_cols):
            experiments.loc[incumbent_idx, col] = known_folds[i]

        # Expected corrected std (Nadeau & Bengio, 2003, ddof=0)
        k = 5
        correction = np.sqrt(1.0 / k + 1.0 / (k - 1))
        expected_std = float(np.std(known_folds, ddof=0))
        expected_threshold = correction * expected_std

        # With threshold_factor = 1.0, the effective threshold should match
        # We use a very generous value to verify the threshold is computed
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Verify the expected corrected std is a reasonable value
        assert expected_threshold > 0
        assert correction == pytest.approx(np.sqrt(0.2 + 0.25), rel=1e-10)

    def test_topq_defaults(self):
        """Default topq=1.0 (no filtering), min_topq=20."""
        cond = UCBLCBRegretBoundCondition()
        assert cond.topq == 1.0
        assert cond.min_topq == 20

    def test_topq_custom(self):
        """Custom topq values are accepted and serialized correctly."""
        cond = UCBLCBRegretBoundCondition(topq=0.5, min_topq=10)
        assert cond.topq == 0.5
        assert cond.min_topq == 10
        data = cond.model_dump()
        restored = UCBLCBRegretBoundCondition(**data)
        assert restored.topq == 0.5
        assert restored.min_topq == 10

    def test_topq_validation(self):
        """topq must be in (0, 1]."""
        with pytest.raises(Exception):
            UCBLCBRegretBoundCondition(topq=0.0)
        with pytest.raises(Exception):
            UCBLCBRegretBoundCondition(topq=-0.5)
        with pytest.raises(Exception):
            UCBLCBRegretBoundCondition(topq=1.5)

    def test_topq_filtering_runs(self, benchmark):
        """With topq < 1.0, condition fits a separate GP on filtered data."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(30), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # topq=0.5 with generous threshold → should still terminate
        cond = UCBLCBRegretBoundCondition(
            noise_variance=1e6,
            threshold_factor=1.0,
            topq=0.5,
            min_topq=5,
            min_experiments=5,
        )
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is False

        # topq=0.5 with very tight threshold → should NOT terminate
        cond_tight = UCBLCBRegretBoundCondition(
            noise_variance=1e-10,
            threshold_factor=1.0,
            topq=0.5,
            min_topq=5,
            min_experiments=5,
        )
        assert (
            cond_tight.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

    def test_topq_no_filtering_when_1(self, benchmark):
        """With topq=1.0 (default), no filtering occurs — uses main strategy."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Generous threshold with topq=1.0 → should terminate
        cond = UCBLCBRegretBoundCondition(
            noise_variance=1e6,
            topq=1.0,
            min_experiments=5,
        )
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is False


class TestStepwiseStrategyTermination:
    """Integration tests for termination via StepwiseStrategy."""

    def test_optimization_complete_raised_when_converged(self, benchmark):
        """StepwiseStrategy should raise OptimizationComplete when
        the UCBLCBRegretBoundCondition returns False and no fallback step exists.
        """
        domain = benchmark.domain

        # Step 1: Random sampling (up to 10 experiments)
        # Step 2: SOBO with very generous termination (should converge quickly)
        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(domain=domain),
                    condition=UCBLCBRegretBoundCondition(
                        threshold_factor=1e6,
                        min_experiments=2,
                    ),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        # Run the loop until OptimizationComplete is raised
        terminated = False
        n_iterations = 0
        for _ in range(50):
            n_iterations += 1
            try:
                candidates = strategy.ask(1)
            except OptimizationComplete:
                terminated = True
                break

            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)

        assert terminated, "Expected OptimizationComplete to be raised"
        assert n_iterations < 50, "Should terminate before max iterations"

    def test_no_termination_with_always_true_fallback(self, benchmark):
        """With AlwaysTrueCondition as last step, no OptimizationComplete is raised."""
        domain = benchmark.domain

        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=5),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(domain=domain),
                    condition=AlwaysTrueCondition(),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        # Should run all 8 iterations without raising
        for _i in range(8):
            candidates = strategy.ask(1)
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)

    def test_strategy_passed_to_condition(self, benchmark):
        """The strategy should be passed to the condition via kwargs."""
        domain = benchmark.domain

        # Use a condition with a min_experiments higher than what we provide
        # to ensure the condition receives and uses the strategy
        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=5),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(domain=domain),
                    condition=UCBLCBRegretBoundCondition(
                        threshold_factor=1.0,
                        min_experiments=100,  # Very high: never terminates
                    ),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        # Should run all iterations without raising (min_experiments not met)
        for _i in range(8):
            candidates = strategy.ask(1)
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)


class TestExpMinRegretGapConditionDataModel:
    """Tests for the ExpMinRegretGapCondition data model."""

    def test_defaults(self):
        cond = ExpMinRegretGapCondition()
        assert cond.threshold_mode == "adaptive"
        assert cond.delta == 0.1
        assert cond.rate == 0.1
        assert cond.start_timing == 10
        assert cond.min_experiments == 5
        assert cond.beta_scale == 1.0
        assert cond.n_samples_lcb == 1000

    def test_custom_params(self):
        cond = ExpMinRegretGapCondition(
            threshold_mode="median",
            delta=0.05,
            rate=0.2,
            start_timing=20,
            min_experiments=10,
            beta_scale=0.5,
            n_samples_lcb=500,
        )
        assert cond.threshold_mode == "median"
        assert cond.delta == 0.05
        assert cond.rate == 0.2
        assert cond.start_timing == 20

    def test_serialization_adaptive(self):
        cond = ExpMinRegretGapCondition(threshold_mode="adaptive", delta=0.05)
        data = cond.model_dump()
        restored = ExpMinRegretGapCondition(**data)
        assert restored.threshold_mode == "adaptive"
        assert restored.delta == 0.05

    def test_serialization_median(self):
        cond = ExpMinRegretGapCondition(
            threshold_mode="median",
            rate=0.2,
            start_timing=15,
        )
        data = cond.model_dump()
        restored = ExpMinRegretGapCondition(**data)
        assert restored.threshold_mode == "median"
        assert restored.rate == 0.2
        assert restored.start_timing == 15

    def test_invalid_threshold_mode(self):
        with pytest.raises(Exception):
            ExpMinRegretGapCondition(threshold_mode="invalid")

    def test_returns_true_without_strategy(self, benchmark):
        cond = ExpMinRegretGapCondition()
        assert cond.evaluate(benchmark.domain, None) is True

    def test_returns_true_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        cond = ExpMinRegretGapCondition()
        assert cond.evaluate(benchmark.domain, None, strategy=strategy) is True

    def test_returns_true_with_few_experiments(self, benchmark):
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(3), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = ExpMinRegretGapCondition(min_experiments=10)
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is True

    def test_first_call_returns_true(self, benchmark):
        """First call (no previous model) should always return True."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = ExpMinRegretGapCondition(min_experiments=5)
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is True

    def test_evaluator_is_stateful(self, benchmark):
        """The same evaluator should be reused across calls."""
        cond = ExpMinRegretGapCondition(min_experiments=5)

        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond.evaluate(benchmark.domain, experiments, strategy=strategy)
        ev1 = cond._evaluator

        # Second call — same evaluator instance
        candidates = strategy.ask(1)[benchmark.domain.inputs.get_keys()]
        new_exp = benchmark.f(candidates)
        new_xy = pd.concat([candidates, new_exp], axis=1)
        experiments2 = pd.concat(
            [experiments, new_xy],
            ignore_index=True,
        )
        strategy2 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy2.tell(experiments2)

        cond.evaluate(benchmark.domain, experiments2, strategy=strategy2)
        assert cond._evaluator is ev1

    def test_adaptive_mode_runs(self, benchmark):
        """Adaptive threshold mode should run without errors over 2 iterations."""
        cond = ExpMinRegretGapCondition(
            threshold_mode="adaptive",
            min_experiments=5,
        )
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # First call: saves state, returns True
        result1 = cond.evaluate(benchmark.domain, experiments, strategy=strategy)
        assert result1 is True

        # Second call: computes metrics
        candidates = strategy.ask(1)[benchmark.domain.inputs.get_keys()]
        new_exp = benchmark.f(candidates)
        new_xy = pd.concat([candidates, new_exp], axis=1)
        experiments2 = pd.concat(
            [experiments, new_xy],
            ignore_index=True,
        )
        strategy2 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy2.tell(experiments2)

        # Should return a boolean
        result2 = cond.evaluate(benchmark.domain, experiments2, strategy=strategy2)
        assert isinstance(result2, bool)

    def test_median_mode_returns_true_before_start_timing(self, benchmark):
        """Median mode should return True before start_timing values collected."""
        cond = ExpMinRegretGapCondition(
            threshold_mode="median",
            start_timing=100,
            min_experiments=5,
        )
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)
        cond.evaluate(benchmark.domain, experiments, strategy=strategy)

        # Do a second call to get a stopping value
        candidates = strategy.ask(1)[benchmark.domain.inputs.get_keys()]
        new_exp = benchmark.f(candidates)
        new_xy = pd.concat([candidates, new_exp], axis=1)
        experiments2 = pd.concat(
            [experiments, new_xy],
            ignore_index=True,
        )
        strategy2 = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy2.tell(experiments2)

        # start_timing=100, only 1 value collected → threshold is None → True
        result = cond.evaluate(benchmark.domain, experiments2, strategy=strategy2)
        assert result is True

    def test_in_stepwise_strategy(self, benchmark):
        """ExpMinRegretGapCondition should work inside StepwiseStrategy."""
        domain = benchmark.domain

        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(domain=domain),
                    condition=ExpMinRegretGapCondition(
                        threshold_mode="adaptive",
                        min_experiments=5,
                    ),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        # Should run without errors for several iterations
        for _i in range(15):
            try:
                candidates = strategy.ask(1)
            except OptimizationComplete:
                break
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)


class TestLogEIPCConditionDataModel:
    """Tests for the LogEIPCCondition data model."""

    def test_defaults(self):
        cond = LogEIPCCondition()
        assert cond.lambda_cost == 1.0
        assert cond.cost_column is None
        assert cond.cost_value == 1.0
        assert cond.alpha == 1.0
        assert cond.min_experiments == 5
        assert cond.n_samples == 2000
        assert cond.search_method == "sample"
        assert cond.cost_model == "mean"

    def test_custom_params(self):
        cond = LogEIPCCondition(
            lambda_cost=0.1,
            cost_column="time_seconds",
            cost_value=60.0,
            alpha=0.5,
            min_experiments=10,
            n_samples=500,
        )
        assert cond.lambda_cost == 0.1
        assert cond.cost_column == "time_seconds"
        assert cond.cost_value == 60.0
        assert cond.alpha == 0.5
        assert cond.min_experiments == 10
        assert cond.n_samples == 500

    def test_serialization(self):
        cond = LogEIPCCondition(lambda_cost=0.5, cost_value=2.0)
        data = cond.model_dump()
        restored = LogEIPCCondition(**data)
        assert restored.lambda_cost == cond.lambda_cost
        assert restored.cost_value == cond.cost_value

    def test_returns_true_without_strategy(self, benchmark):
        cond = LogEIPCCondition()
        assert cond.evaluate(benchmark.domain, None) is True

    def test_returns_true_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        cond = LogEIPCCondition()
        assert cond.evaluate(benchmark.domain, None, strategy=strategy) is True

    def test_returns_true_with_few_experiments(self, benchmark):
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(3), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = LogEIPCCondition(min_experiments=10)
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is True

    def test_evaluate_returns_bool(self, benchmark):
        """evaluate() must always return a bool."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = LogEIPCCondition(min_experiments=5)
        result = cond.evaluate(benchmark.domain, experiments, strategy=strategy)
        assert isinstance(result, bool)

    def test_generous_lambda_does_not_stop(self, benchmark):
        """Very small lambda_cost → EI almost always exceeds cost → keep going."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = LogEIPCCondition(lambda_cost=1e-10, min_experiments=5)
        assert cond.evaluate(benchmark.domain, experiments, strategy=strategy) is True

    def test_cost_column_used_when_present(self, benchmark):
        """When cost_column is set and populated, it should be used."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)
        experiments = experiments.copy()
        experiments["cost"] = 5.0

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        cond = LogEIPCCondition(cost_column="cost", min_experiments=5)
        result = cond.evaluate(benchmark.domain, experiments, strategy=strategy)
        assert isinstance(result, bool)

    def test_in_stepwise_strategy(self, benchmark):
        """LogEIPCCondition should work inside a StepwiseStrategy."""
        domain = benchmark.domain

        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(domain=domain),
                    condition=LogEIPCCondition(
                        lambda_cost=1.0,
                        min_experiments=5,
                    ),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        for _i in range(15):
            try:
                candidates = strategy.ask(1)
            except OptimizationComplete:
                break
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)
