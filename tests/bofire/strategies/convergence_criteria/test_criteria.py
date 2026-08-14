"""Tests for the GP-based convergence criteria.

Covers the criterion data models (defaults, validation, serialization), the
functional ``evaluate_*_criterion`` evaluators (guards and stop/continue
decisions), ``strategy.has_converged()`` integration, and the stepwise bridge
via ``StrategyHasConvergedCondition``.
"""

import numpy as np
import pytest

from bofire.benchmarks.multi import DTLZ2
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.strategies.api import MoboStrategy as MoboStrategyDataModel
from bofire.data_models.strategies.api import (
    NumberOfExperimentsCondition,
    Step,
    StrategyHasConvergedCondition,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.data_models.strategies.api import (
    StepwiseStrategy as StepwiseStrategyDataModel,
)
from bofire.data_models.strategies.convergence_criteria.api import (
    ExpMinRegretGapCriterion,
    LogEipcCriterion,
    ProbabilisticRegretBoundCriterion,
    UcbLcbRegretBoundCriterion,
)
from bofire.strategies.api import RandomStrategy, SoboStrategy, StepwiseStrategy
from bofire.strategies.convergence_criteria.exp_min_regret_gap import (
    evaluate_exp_min_regret_gap_criterion,
)
from bofire.strategies.convergence_criteria.log_eipc import evaluate_log_eipc_criterion
from bofire.strategies.convergence_criteria.probabilistic_regret_bound import (
    evaluate_probabilistic_regret_bound_criterion,
)
from bofire.strategies.convergence_criteria.ucb_lcb import (
    evaluate_ucb_lcb_regret_bound_criterion,
)


GP_CRITERIA = [
    UcbLcbRegretBoundCriterion,
    ExpMinRegretGapCriterion,
    LogEipcCriterion,
    ProbabilisticRegretBoundCriterion,
]


@pytest.fixture
def benchmark():
    return Himmelblau()


def _fitted_sobo(benchmark, n=10, criterion=None):
    """Return a SoboStrategy fitted on ``n`` random experiments."""
    random = RandomStrategy(data_model=RandomStrategyDataModel(domain=benchmark.domain))
    experiments = benchmark.f(random.ask(n), return_complete=True)
    strategy = SoboStrategy(
        data_model=SoboStrategyDataModel(
            domain=benchmark.domain, convergence_criterion=criterion
        )
    )
    strategy.tell(experiments)
    return strategy, experiments


class TestApplicability:
    """All four GP-based criteria are single-objective only."""

    @pytest.mark.parametrize("criterion_cls", GP_CRITERIA)
    def test_applicability_flags(self, criterion_cls):
        assert criterion_cls.is_applicable_to_singleobjective() is True
        assert criterion_cls.is_applicable_to_multiobjective() is False
        assert criterion_cls.is_applicable_to_objective_free() is False

    @pytest.mark.parametrize("criterion_cls", GP_CRITERIA)
    def test_accepted_by_sobo(self, criterion_cls, benchmark):
        data_model = SoboStrategyDataModel(
            domain=benchmark.domain, convergence_criterion=criterion_cls()
        )
        assert isinstance(data_model.convergence_criterion, criterion_cls)

    @pytest.mark.parametrize("criterion_cls", GP_CRITERIA)
    def test_rejected_by_mobo(self, criterion_cls):
        domain = DTLZ2(dim=3).domain
        with pytest.raises(ValueError, match="not implemented for strategy"):
            MoboStrategyDataModel(domain=domain, convergence_criterion=criterion_cls())


class TestInvalidExperimentsIgnored:
    """Rows with invalid outputs are excluded from the convergence checks.

    The surrogate is only fit on valid experiments, so the criteria must not
    compute incumbents, best values, or thresholds from invalidated rows.
    """

    @staticmethod
    def _fitted_sobo_with_invalid(benchmark, n_valid, criterion):
        import pandas as pd

        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(n_valid + 2), return_complete=True)
        # Invalidate the two rows with the *best* objective values: if these
        # leaked into the checks, incumbent-based quantities would change.
        worst_first = experiments["y"].sort_values().index[:2]
        experiments.loc[worst_first, "valid_y"] = 0
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(
                domain=benchmark.domain, convergence_criterion=criterion
            )
        )
        strategy.tell(experiments)
        assert isinstance(strategy.experiments, pd.DataFrame)
        return strategy

    def test_invalid_rows_do_not_count_towards_min_experiments(self, benchmark):
        criterion = UcbLcbRegretBoundCriterion(min_experiments=9)
        # 7 valid + 2 invalid rows: below min_experiments -> not converged.
        strategy = self._fitted_sobo_with_invalid(benchmark, 7, criterion)
        assert evaluate_ucb_lcb_regret_bound_criterion(criterion, strategy) is False

    @pytest.mark.parametrize("criterion_cls", GP_CRITERIA)
    def test_criteria_run_with_invalid_rows_present(self, benchmark, criterion_cls):
        kwargs = {"min_experiments": 5}
        if criterion_cls is ProbabilisticRegretBoundCriterion:
            kwargs.update(n_samples_max=32, n_random=64, n_starts=2)
        if criterion_cls is ExpMinRegretGapCriterion:
            kwargs.update(n_samples_lcb=100)
        criterion = criterion_cls(**kwargs)
        strategy = self._fitted_sobo_with_invalid(benchmark, 10, criterion)
        assert isinstance(strategy.has_converged(), bool)


class TestUcbLcbRegretBoundCriterion:
    """Tests for the criterion data model and its evaluator."""

    def test_defaults(self):
        criterion = UcbLcbRegretBoundCriterion()
        assert criterion.noise_variance is None
        assert criterion.threshold_factor == 1.0
        assert criterion.min_experiments == 5
        assert criterion.topq == 0.5
        assert criterion.min_topq == 20

    def test_custom_params(self):
        criterion = UcbLcbRegretBoundCriterion(
            noise_variance=0.1,
            threshold_factor=2.0,
            min_experiments=10,
            topq=0.5,
            min_topq=10,
        )
        assert criterion.noise_variance == 0.1
        assert criterion.threshold_factor == 2.0
        assert criterion.min_experiments == 10
        assert criterion.topq == 0.5
        assert criterion.min_topq == 10

    def test_serialization(self):
        criterion = UcbLcbRegretBoundCriterion(noise_variance=0.1, threshold_factor=2.0)
        data = criterion.model_dump()
        restored = UcbLcbRegretBoundCriterion(**data)
        assert restored == criterion

    def test_cv_mode_validation(self):
        """noise_variance='cv' requires cv_fold_columns with >= 2 columns."""
        with pytest.raises(ValueError, match="cv_fold_columns"):
            UcbLcbRegretBoundCriterion(noise_variance="cv")

        with pytest.raises(ValueError, match="cv_fold_columns"):
            UcbLcbRegretBoundCriterion(noise_variance="cv", cv_fold_columns=["fold_0"])

        criterion = UcbLcbRegretBoundCriterion(
            noise_variance="cv",
            cv_fold_columns=["fold_0", "fold_1", "fold_2"],
        )
        assert criterion.noise_variance == "cv"
        assert len(criterion.cv_fold_columns) == 3

    def test_cv_mode_serialization(self):
        criterion = UcbLcbRegretBoundCriterion(
            noise_variance="cv",
            cv_fold_columns=["f0", "f1", "f2", "f3", "f4"],
            threshold_factor=0.5,
        )
        restored = UcbLcbRegretBoundCriterion(**criterion.model_dump())
        assert restored == criterion

    def test_topq_validation(self):
        """topq must be in (0, 1]."""
        with pytest.raises(Exception):
            UcbLcbRegretBoundCriterion(topq=0.0)
        with pytest.raises(Exception):
            UcbLcbRegretBoundCriterion(topq=-0.5)
        with pytest.raises(Exception):
            UcbLcbRegretBoundCriterion(topq=1.5)

    def test_not_converged_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        criterion = UcbLcbRegretBoundCriterion()
        assert evaluate_ucb_lcb_regret_bound_criterion(criterion, strategy) is False

    def test_not_converged_with_few_experiments(self, benchmark):
        strategy, _ = _fitted_sobo(benchmark, n=3)
        criterion = UcbLcbRegretBoundCriterion(min_experiments=10)
        assert evaluate_ucb_lcb_regret_bound_criterion(criterion, strategy) is False

    def test_evaluates_regret_bound(self, benchmark):
        """With a fitted strategy and enough data, the regret bound decides."""
        # Very tight threshold (manual noise_variance mode): not converged.
        strategy, _ = _fitted_sobo(
            benchmark,
            criterion=UcbLcbRegretBoundCriterion(
                noise_variance=1e-10, topq=1.0, min_experiments=5
            ),
        )
        assert strategy.has_converged() is False

        # Very generous threshold (manual noise_variance mode): converged.
        strategy, _ = _fitted_sobo(
            benchmark,
            criterion=UcbLcbRegretBoundCriterion(
                noise_variance=1e6, topq=1.0, min_experiments=5
            ),
        )
        assert strategy.has_converged() is True

    def test_gp_noise_threshold(self, benchmark):
        """Default (noise_variance=None) uses GP estimated noise."""
        strategy, _ = _fitted_sobo(benchmark)

        # Very large threshold_factor with GP noise → converged.
        criterion_generous = UcbLcbRegretBoundCriterion(
            threshold_factor=1e8, topq=1.0, min_experiments=5
        )
        assert criterion_generous.noise_variance is None  # GP noise mode
        assert (
            evaluate_ucb_lcb_regret_bound_criterion(criterion_generous, strategy)
            is True
        )

        # Tiny threshold_factor → threshold ~ 0 → not converged.
        criterion_strict = UcbLcbRegretBoundCriterion(
            threshold_factor=1e-12, topq=1.0, min_experiments=5
        )
        assert (
            evaluate_ucb_lcb_regret_bound_criterion(criterion_strict, strategy) is False
        )

    def test_cv_mode_threshold(self, benchmark):
        """With noise_variance='cv', uses incumbent's CV fold std for threshold."""
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        experiments = benchmark.f(random.ask(10), return_complete=True)

        # Add synthetic fold columns with known variability.
        n_folds = 5
        fold_cols = [f"y_fold_{i}" for i in range(n_folds)]
        rng = np.random.RandomState(42)
        for col in fold_cols:
            experiments[col] = experiments["y"] + rng.normal(0, 0.5, len(experiments))

        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        strategy.tell(experiments)

        # Very large threshold_factor → threshold >> regret bound → converged.
        criterion_generous = UcbLcbRegretBoundCriterion(
            noise_variance="cv",
            cv_fold_columns=fold_cols,
            threshold_factor=1e6,
            topq=1.0,
            min_experiments=5,
        )
        assert (
            evaluate_ucb_lcb_regret_bound_criterion(criterion_generous, strategy)
            is True
        )

        # Very small threshold_factor → not converged.
        criterion_tight = UcbLcbRegretBoundCriterion(
            noise_variance="cv",
            cv_fold_columns=fold_cols,
            threshold_factor=1e-10,
            topq=1.0,
            min_experiments=5,
        )
        assert (
            evaluate_ucb_lcb_regret_bound_criterion(criterion_tight, strategy) is False
        )

    def test_topq_filtering_runs(self, benchmark):
        """With topq < 1.0, the evaluator fits a separate GP on filtered data."""
        strategy, _ = _fitted_sobo(benchmark, n=30)

        criterion = UcbLcbRegretBoundCriterion(
            noise_variance=1e6,
            threshold_factor=1.0,
            topq=0.5,
            min_topq=5,
            min_experiments=5,
        )
        assert evaluate_ucb_lcb_regret_bound_criterion(criterion, strategy) is True

        criterion_tight = UcbLcbRegretBoundCriterion(
            noise_variance=1e-10,
            threshold_factor=1.0,
            topq=0.5,
            min_topq=5,
            min_experiments=5,
        )
        assert (
            evaluate_ucb_lcb_regret_bound_criterion(criterion_tight, strategy) is False
        )


class TestUcbLcbTopQDirection:
    """Top-q filtering in the UCB-LCB criterion is objective-aware.

    It refits the regret-bound GP on the best fraction of observations.  For
    minimisation the "best" are the lowest-y points; for maximisation the
    highest-y.  Minimising ``y`` and maximising ``-y`` are the same problem, so
    the criterion must reach the same convergence decision in both framings.
    """

    @staticmethod
    def _fit(objective, X, y, seed=0):
        import pandas as pd
        import torch

        from bofire.data_models.domain.api import Domain, Inputs, Outputs
        from bofire.data_models.features.api import ContinuousInput, ContinuousOutput

        torch.manual_seed(seed)
        domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x{i}", bounds=(0.0, 1.0))
                    for i in range(X.shape[1])
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y", objective=objective)]),
        )
        exp = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        exp["y"] = y
        exp["valid_y"] = 1
        strat = SoboStrategy(data_model=SoboStrategyDataModel(domain=domain))
        strat.tell(exp)
        return strat

    def test_topq_negation_invariance_decision(self):
        import torch

        from bofire.data_models.objectives.api import (
            MaximizeObjective,
            MinimizeObjective,
        )

        rng = np.random.default_rng(3)
        X = rng.random((12, 2))
        y = (X[:, 0] - 0.3) ** 2 + 0.5 * (X[:, 1] - 0.6) ** 2

        criterion = UcbLcbRegretBoundCriterion(
            topq=0.5,
            min_topq=3,
            min_experiments=3,
            noise_variance=1e-3,
            threshold_factor=1.0,
        )

        strat_min = self._fit(MinimizeObjective(), X, y)
        strat_max = self._fit(MaximizeObjective(), X, -y)

        torch.manual_seed(11)
        dec_min = evaluate_ucb_lcb_regret_bound_criterion(criterion, strat_min)
        torch.manual_seed(11)
        dec_max = evaluate_ucb_lcb_regret_bound_criterion(criterion, strat_max)

        assert isinstance(dec_min, bool) and isinstance(dec_max, bool)
        assert dec_min == dec_max  # top-q picked the equivalent rows in both


class TestExpMinRegretGapCriterion:
    """Tests for the ExpMinRegretGapCriterion data model and its evaluator."""

    def test_defaults(self):
        criterion = ExpMinRegretGapCriterion()
        assert criterion.threshold_mode == "adaptive"
        assert criterion.delta == 0.1
        assert criterion.rate == 0.1
        assert criterion.start_timing == 10
        assert criterion.min_experiments == 5
        assert criterion.beta_scale == 1.0
        assert criterion.n_samples_lcb == 1000

    def test_custom_params(self):
        criterion = ExpMinRegretGapCriterion(
            threshold_mode="median",
            delta=0.05,
            rate=0.2,
            start_timing=20,
            min_experiments=10,
            beta_scale=0.5,
            n_samples_lcb=500,
        )
        assert criterion.threshold_mode == "median"
        assert criterion.delta == 0.05
        assert criterion.rate == 0.2
        assert criterion.start_timing == 20

    def test_serialization(self):
        criterion = ExpMinRegretGapCriterion(
            threshold_mode="median",
            rate=0.2,
            start_timing=15,
            noise_var_override=1e-6,
        )
        restored = ExpMinRegretGapCriterion(**criterion.model_dump())
        assert restored == criterion

    def test_invalid_threshold_mode(self):
        with pytest.raises(Exception):
            ExpMinRegretGapCriterion(threshold_mode="invalid")

    def test_not_converged_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        criterion = ExpMinRegretGapCriterion()
        assert evaluate_exp_min_regret_gap_criterion(criterion, strategy) is False

    def test_not_converged_with_few_experiments(self, benchmark):
        strategy, _ = _fitted_sobo(benchmark, n=3)
        criterion = ExpMinRegretGapCriterion(min_experiments=10)
        assert evaluate_exp_min_regret_gap_criterion(criterion, strategy) is False

    def test_adaptive_mode_runs(self, benchmark):
        """Adaptive threshold mode reconstructs the previous model and decides."""
        criterion = ExpMinRegretGapCriterion(
            threshold_mode="adaptive",
            min_experiments=5,
            n_samples_lcb=200,
        )
        strategy, _ = _fitted_sobo(benchmark, criterion=criterion)
        result = strategy.has_converged()
        assert isinstance(result, bool)

    def test_median_mode_not_converged_before_start_timing(self, benchmark):
        """Median mode cannot trigger before start_timing values exist."""
        criterion = ExpMinRegretGapCriterion(
            threshold_mode="median",
            start_timing=100,
            min_experiments=5,
        )
        strategy, _ = _fitted_sobo(benchmark, criterion=criterion)
        assert strategy.has_converged() is False

    def test_median_mode_replays_history(self, benchmark):
        """Median mode replays early stopping values from history prefixes."""
        criterion = ExpMinRegretGapCriterion(
            threshold_mode="median",
            start_timing=2,
            min_experiments=3,
            n_samples_lcb=100,
        )
        strategy, _ = _fitted_sobo(benchmark, n=8, criterion=criterion)
        result = strategy.has_converged()
        assert isinstance(result, bool)

    def test_warm_cache_avoids_refits(self, benchmark, monkeypatch):
        """Consecutive checks in a running loop reuse the previous model.

        Only the cold start reconstructs the previous-iteration model by
        refitting; repeated checks at the same experiment count return the
        cached decision, and a check after exactly one new experiment takes
        the warm path with zero extra fits.
        """
        import bofire.strategies.convergence_criteria.exp_min_regret_gap as emrg

        refits = {"n": 0}
        original = emrg._refit_on_prefix

        def counting(*args, **kwargs):
            refits["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(emrg, "_refit_on_prefix", counting)

        criterion = ExpMinRegretGapCriterion(min_experiments=3, n_samples_lcb=100)
        strategy, _ = _fitted_sobo(benchmark, n=5, criterion=criterion)

        # Cold start: exactly one refit (previous model on n-1 experiments).
        first = strategy.has_converged()
        assert refits["n"] == 1

        # Repeated check without new data: cached decision, no extra work.
        assert strategy.has_converged() is first
        assert refits["n"] == 1

        # One new experiment: warm path reuses the cached previous model.
        random = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain)
        )
        new = benchmark.f(random.ask(1), return_complete=True)
        strategy.tell(new)
        assert isinstance(strategy.has_converged(), bool)
        assert refits["n"] == 1


class TestLogEipcCriterion:
    """Tests for the LogEipcCriterion data model and its evaluator."""

    def test_defaults(self):
        criterion = LogEipcCriterion()
        assert criterion.lambda_cost == 1.0
        assert criterion.cost_column is None
        assert criterion.cost_value == 1.0
        assert criterion.alpha == 1.0
        assert criterion.min_experiments == 5
        assert criterion.n_samples == 2000
        assert criterion.search_method == "sample"
        assert criterion.cost_model == "mean"

    def test_custom_params(self):
        criterion = LogEipcCriterion(
            lambda_cost=0.1,
            cost_column="time_seconds",
            cost_value=60.0,
            alpha=0.5,
            min_experiments=10,
            n_samples=500,
        )
        assert criterion.lambda_cost == 0.1
        assert criterion.cost_column == "time_seconds"
        assert criterion.cost_value == 60.0
        assert criterion.alpha == 0.5
        assert criterion.min_experiments == 10
        assert criterion.n_samples == 500

    def test_serialization(self):
        criterion = LogEipcCriterion(lambda_cost=0.5, cost_value=2.0)
        restored = LogEipcCriterion(**criterion.model_dump())
        assert restored == criterion

    def test_not_converged_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        criterion = LogEipcCriterion()
        assert evaluate_log_eipc_criterion(criterion, strategy) is False

    def test_not_converged_with_few_experiments(self, benchmark):
        strategy, _ = _fitted_sobo(benchmark, n=3)
        criterion = LogEipcCriterion(min_experiments=10)
        assert evaluate_log_eipc_criterion(criterion, strategy) is False

    def test_evaluate_returns_bool(self, benchmark):
        criterion = LogEipcCriterion(min_experiments=5)
        strategy, _ = _fitted_sobo(benchmark, criterion=criterion)
        assert isinstance(strategy.has_converged(), bool)

    def test_generous_lambda_does_not_converge(self, benchmark):
        """Very small lambda_cost → EI almost always exceeds cost → not converged."""
        criterion = LogEipcCriterion(lambda_cost=1e-10, min_experiments=5)
        strategy, _ = _fitted_sobo(benchmark, criterion=criterion)
        assert strategy.has_converged() is False

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

        criterion = LogEipcCriterion(cost_column="cost", min_experiments=5)
        result = evaluate_log_eipc_criterion(criterion, strategy)
        assert isinstance(result, bool)


class TestProbabilisticRegretBoundCriterion:
    """Tests for ProbabilisticRegretBoundCriterion and its evaluator."""

    def test_defaults(self):
        criterion = ProbabilisticRegretBoundCriterion()
        assert criterion.epsilon is None
        assert criterion.epsilon_relative == pytest.approx(0.01)
        assert criterion.delta_mod == pytest.approx(0.05)
        assert criterion.delta_est == pytest.approx(0.05)
        assert criterion.enforce_convergence is True
        assert criterion.n_samples_max == 1024
        assert criterion.min_experiments == 5
        assert criterion.n_starts == 8
        assert criterion.n_random == 512
        assert criterion.n_test_points == 1

    def test_custom_params(self):
        criterion = ProbabilisticRegretBoundCriterion(
            epsilon=0.5,
            delta_mod=0.03,
            delta_est=0.03,
            enforce_convergence=False,
            n_samples_max=256,
            min_experiments=10,
            n_starts=4,
            n_random=128,
            n_test_points=3,
        )
        assert criterion.epsilon == pytest.approx(0.5)
        assert criterion.delta_mod == pytest.approx(0.03)
        assert criterion.delta_est == pytest.approx(0.03)
        assert criterion.enforce_convergence is False
        assert criterion.n_samples_max == 256
        assert criterion.min_experiments == 10
        assert criterion.n_starts == 4
        assert criterion.n_random == 128
        assert criterion.n_test_points == 3

    def test_serialization_roundtrip(self):
        criterion = ProbabilisticRegretBoundCriterion(
            epsilon=1.0,
            delta_mod=0.1,
            delta_est=0.1,
            n_samples_max=512,
        )
        restored = ProbabilisticRegretBoundCriterion(**criterion.model_dump())
        assert restored == criterion

    def test_type_literal(self):
        criterion = ProbabilisticRegretBoundCriterion()
        assert criterion.type == "ProbabilisticRegretBoundCriterion"

    def test_epsilon_none_accepted(self):
        criterion = ProbabilisticRegretBoundCriterion(epsilon=None)
        assert criterion.epsilon is None

    def test_delta_boundary_rejected(self):
        with pytest.raises(Exception):
            ProbabilisticRegretBoundCriterion(delta_mod=0.0)
        with pytest.raises(Exception):
            ProbabilisticRegretBoundCriterion(delta_mod=1.0)
        with pytest.raises(Exception):
            ProbabilisticRegretBoundCriterion(delta_est=0.0)
        with pytest.raises(Exception):
            ProbabilisticRegretBoundCriterion(delta_est=1.0)

    def test_not_converged_with_unfitted_strategy(self, benchmark):
        strategy = SoboStrategy(
            data_model=SoboStrategyDataModel(domain=benchmark.domain)
        )
        criterion = ProbabilisticRegretBoundCriterion()
        assert (
            evaluate_probabilistic_regret_bound_criterion(criterion, strategy) is False
        )

    def test_not_converged_below_min_experiments(self, benchmark):
        strategy, _ = _fitted_sobo(benchmark, n=3)
        criterion = ProbabilisticRegretBoundCriterion(min_experiments=10)
        assert (
            evaluate_probabilistic_regret_bound_criterion(criterion, strategy) is False
        )

    def test_evaluate_returns_bool(self, benchmark):
        criterion = ProbabilisticRegretBoundCriterion(
            n_samples_max=32,
            n_random=64,
            n_starts=2,
            min_experiments=5,
        )
        strategy, _ = _fitted_sobo(benchmark, criterion=criterion)
        assert isinstance(strategy.has_converged(), bool)

    def test_huge_epsilon_converges(self, benchmark):
        """ε >> any plausible regret → criterion satisfied → converged."""
        strategy, experiments = _fitted_sobo(benchmark)

        output_key = benchmark.domain.outputs.get_keys()[0]
        y_range = float(experiments[output_key].max() - experiments[output_key].min())

        criterion = ProbabilisticRegretBoundCriterion(
            epsilon=100.0 * y_range,
            delta_mod=0.495,
            delta_est=0.495,
            enforce_convergence=False,
            n_samples_max=32,
            n_random=64,
            n_starts=2,
            min_experiments=5,
        )
        assert (
            evaluate_probabilistic_regret_bound_criterion(criterion, strategy) is True
        )

    def test_tiny_epsilon_does_not_converge(self, benchmark):
        """Negligibly small ε → regret rarely ≤ ε → not converged."""
        strategy, _ = _fitted_sobo(benchmark)

        # epsilon_relative=1e-10 gives ε ≈ 1e-8 (floored), far below any
        # real regret → P(regret ≤ ε) ≈ 0 → CI converges above level.
        criterion = ProbabilisticRegretBoundCriterion(
            epsilon=None,
            epsilon_relative=1e-10,
            delta_mod=0.1,
            delta_est=0.1,
            enforce_convergence=True,
            n_samples_max=64,
            n_random=64,
            n_starts=2,
            min_experiments=5,
        )
        assert (
            evaluate_probabilistic_regret_bound_criterion(criterion, strategy) is False
        )


class TestStepwiseStrategyConvergence:
    """Integration tests for convergence via the StrategyHasConvergedCondition."""

    def test_stepwise_stops_when_converged(self, benchmark):
        """Once the SOBO step's criterion is met and no fallback step exists,
        the StepwiseStrategy raises because no condition is satisfied.
        """
        domain = benchmark.domain

        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=10),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(
                        domain=domain,
                        convergence_criterion=UcbLcbRegretBoundCriterion(
                            noise_variance=1e6,
                            topq=1.0,
                            min_experiments=2,
                        ),
                    ),
                    condition=StrategyHasConvergedCondition(),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        terminated = False
        n_iterations = 0
        for _ in range(50):
            n_iterations += 1
            try:
                candidates = strategy.ask(1)
            except ValueError:
                terminated = True
                break

            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)

        assert terminated, "Expected the stepwise strategy to run out of steps"
        assert n_iterations < 50, "Should terminate before max iterations"

    def test_stepwise_keeps_running_when_not_converged(self, benchmark):
        """With min_experiments never reached, the SOBO step stays active."""
        domain = benchmark.domain

        data_model = StepwiseStrategyDataModel(
            domain=domain,
            steps=[
                Step(
                    strategy_data=RandomStrategyDataModel(domain=domain),
                    condition=NumberOfExperimentsCondition(n_experiments=5),
                ),
                Step(
                    strategy_data=SoboStrategyDataModel(
                        domain=domain,
                        convergence_criterion=UcbLcbRegretBoundCriterion(
                            min_experiments=100,  # Very high: never converges
                        ),
                    ),
                    condition=StrategyHasConvergedCondition(),
                ),
            ],
        )
        strategy = StepwiseStrategy(data_model=data_model)

        # Should run all 8 iterations without raising.
        for _i in range(8):
            candidates = strategy.ask(1)
            candidates = candidates[domain.inputs.get_keys()]
            experiments = benchmark.f(candidates, return_complete=True)
            strategy.tell(experiments)
