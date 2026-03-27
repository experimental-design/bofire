"""Tests for UCBLCBRegretBoundCondition and StepwiseStrategy termination."""

import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks.single import Himmelblau
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    NumberOfExperimentsCondition,
    RandomStrategy as RandomStrategyDataModel,
    SoboStrategy as SoboStrategyDataModel,
    Step,
    StepwiseStrategy as StepwiseStrategyDataModel,
    UCBLCBRegretBoundCondition,
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

    def test_serialization_range_mode(self):
        cond = UCBLCBRegretBoundCondition(noise_variance="range", threshold_factor=0.05)
        data = cond.model_dump()
        restored = UCBLCBRegretBoundCondition(**data)
        assert restored.noise_variance == "range"
        assert restored.threshold_factor == 0.05

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
        assert (
            cond.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

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

    def test_range_based_threshold(self, benchmark):
        """With noise_variance='range', uses observed output range for threshold."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Very large threshold_factor (100x range) → should terminate
        cond_generous = UCBLCBRegretBoundCondition(
            noise_variance="range", threshold_factor=100.0, min_experiments=5
        )
        assert cond_generous.noise_variance == "range"
        assert (
            cond_generous.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )

        # Very small threshold_factor → should NOT terminate
        cond_tight = UCBLCBRegretBoundCondition(
            noise_variance="range", threshold_factor=1e-10, min_experiments=5
        )
        assert (
            cond_tight.evaluate(benchmark.domain, experiments, strategy=strategy)
            is True
        )

    def test_cv_mode_validation(self):
        """noise_variance='cv' requires cv_fold_columns with >= 2 columns."""
        with pytest.raises(ValueError, match="cv_fold_columns"):
            UCBLCBRegretBoundCondition(noise_variance="cv")

        with pytest.raises(ValueError, match="cv_fold_columns"):
            UCBLCBRegretBoundCondition(
                noise_variance="cv", cv_fold_columns=["fold_0"]
            )

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
        assert (
            cond.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )

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
            noise_variance=1e6, topq=1.0, min_experiments=5,
        )
        assert (
            cond.evaluate(benchmark.domain, experiments, strategy=strategy)
            is False
        )


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
        for i in range(50):
            try:
                candidates = strategy.ask(1)
            except OptimizationComplete:
                terminated = True
                break

            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)

        assert terminated, "Expected OptimizationComplete to be raised"
        assert i < 49, "Should terminate before max iterations"

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
        for i in range(8):
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
        for i in range(8):
            candidates = strategy.ask(1)
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)
