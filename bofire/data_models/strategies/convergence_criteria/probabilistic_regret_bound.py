from typing import Annotated, Literal

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

    Reference:
        Wilson (2024): "Stopping Bayesian Optimization with Probabilistic
            Regret Bounds" (NeurIPS 2024).
    """

    type: Literal["ProbabilisticRegretBoundCriterion"] = (
        "ProbabilisticRegretBoundCriterion"
    )
    epsilon: PositiveFloat | None = Field(
        default=None,
        description="Absolute simple-regret threshold in objective units; if "
        "None, derived from `epsilon_relative`.",
    )
    epsilon_relative: Annotated[float, Field(gt=0, le=1)] = Field(
        default=0.01,
        description="Fractional regret threshold relative to the observed "
        "objective range; ignored when `epsilon` is set.",
    )
    delta_mod: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.05,
        description="Model-risk tolerance; converged when the estimated "
        "probability of the regret exceeding epsilon falls below it.",
    )
    delta_est: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.05,
        description="Estimation-risk tolerance of the sequential "
        "Clopper-Pearson test.",
    )
    optim_method: str = Field(
        default="L-BFGS-B",
        description="scipy optimization method for path minimization.",
    )
    optim_maxiter: PositiveInt = Field(
        default=200,
        description="Maximum iterations per optimization start.",
    )
    optim_ftol: Annotated[float, Field(gt=0)] = Field(
        default=1e-9,
        description="Function-value convergence tolerance for path " "minimization.",
    )
    enforce_convergence: bool = Field(
        default=True,
        description="Report convergence only when the Clopper-Pearson interval "
        "conclusively excludes `delta_mod`; False uses the raw Monte Carlo "
        "estimate.",
    )
    n_samples_max: PositiveInt = Field(
        default=1024,
        description="Maximum GP path samples per convergence check.",
    )
    initial_batch: PositiveInt = Field(
        default=16,
        description="Initial cumulative sample target of the Clopper-Pearson " "test.",
    )
    batch_growth: Annotated[float, Field(gt=1.0)] = Field(
        default=1.5,
        description="Geometric growth factor of the cumulative sample " "schedule.",
    )
    min_experiments: PositiveInt = Field(
        default=5,
        description="Minimum experiments before convergence is checked.",
    )
    n_starts: PositiveInt = Field(
        default=8,
        description="Local-optimization starts per sample path.",
    )
    n_random: PositiveInt = Field(
        default=512,
        description="Random domain points for selecting optimization starts.",
    )
    n_test_points: PositiveInt = Field(
        default=1,
        description="Candidate points the criterion is tested at; 1 tests the "
        "incumbent only, larger values add the best in-sample points under "
        "the posterior mean.",
    )

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
