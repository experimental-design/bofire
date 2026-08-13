from typing import Annotated, Literal, Optional

from pydantic import Field, PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class ProbabilisticRegretBoundCriterion(ConvergenceCriterion):
    """Convergence based on probabilistic regret bounds (Wilson, 2024).

    The optimization is considered converged once the Clopper-Pearson
    sequential hypothesis test certifies that the estimated probability of the
    incumbent's regret exceeding ε has dropped to the model-risk threshold,
    ``P̂(regret > ε) ≤ δ_mod``.

    The two risk parameters ``delta_mod`` and ``delta_est`` correspond directly
    to the paper's δ_mod (model risk) and δ_est (estimation error from the
    Monte Carlo test).

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``); the evaluator
    draws posterior sample paths from the strategy's model.
    Single-objective only.

    Attributes:
        epsilon: Absolute simple regret threshold in Y units.  If ``None``
            (default), computed from ``epsilon_relative``.
        epsilon_relative: Fractional ε relative to the observed Y range.
            Default ``0.01`` (1 %).  Ignored when ``epsilon`` is set.
        delta_mod: Model-risk tolerance δ_mod.  Convergence triggers when the
            estimated probability that regret exceeds ε falls below this
            value.  Default ``0.05``.
        delta_est: Estimation-risk tolerance δ_est for the sequential
            Clopper-Pearson test.  Default ``0.05``.
        enforce_convergence: Only report convergence when the CP CI
            conclusively excludes δ_mod (default ``True``).  Set to ``False``
            to use the raw MC estimate.
        n_samples_max: Maximum GP path samples per convergence check.
            Default ``1024``.
        initial_batch: Initial cumulative sample target for the Clopper-Pearson
            level test.  Default ``16``.
        batch_growth: Geometric growth factor for the cumulative sample
            schedule (must be ``> 1``).  Default ``1.5``.
        min_experiments: Minimum experiments before convergence is checked.
            Default ``5``.
        n_starts: L-BFGS-B starts per path for path minimization.  Default ``8``.
        n_random: Random domain points for identifying L-BFGS-B start
            candidates.  Default ``512``.
        n_test_points: Number of candidate points to evaluate the criterion
            at.  ``1`` (default) tests the incumbent only; values ``> 1`` also
            include the ``n_test_points − 1`` in-sample points that are best
            under the GP posterior mean.
        optim_method: scipy optimisation method for path minimization.
            Default ``"L-BFGS-B"``.
        optim_maxiter: Maximum iterations per optimisation start.
            Default ``200``.
        optim_ftol: Function-value convergence tolerance for path
            minimization.  Default ``1e-9``.

    Reference:
        Wilson (2024): "Stopping Bayesian Optimization with Probabilistic
            Regret Bounds" (NeurIPS 2024).
    """

    type: Literal["ProbabilisticRegretBoundCriterion"] = (
        "ProbabilisticRegretBoundCriterion"
    )
    epsilon: Optional[PositiveFloat] = None
    epsilon_relative: Annotated[float, Field(gt=0, le=1)] = 0.01
    delta_mod: Annotated[float, Field(gt=0, lt=1)] = 0.05
    delta_est: Annotated[float, Field(gt=0, lt=1)] = 0.05
    optim_method: str = "L-BFGS-B"
    optim_maxiter: PositiveInt = 200
    optim_ftol: Annotated[float, Field(gt=0)] = 1e-9
    enforce_convergence: bool = True
    n_samples_max: PositiveInt = 1024
    initial_batch: PositiveInt = 16
    batch_growth: Annotated[float, Field(gt=1.0)] = 1.5
    min_experiments: PositiveInt = 5
    n_starts: PositiveInt = 8
    n_random: PositiveInt = 512
    n_test_points: PositiveInt = 1

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
