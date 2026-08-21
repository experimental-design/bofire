from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class ExpMinRegretGapCriterion(ConvergenceCriterion):
    """Expected minimum regret gap criterion (Ishibashi et al. 2023).

    The optimization is considered converged once the stopping value
    ``delta_f + ei_diff + kappa * sqrt(KL / 2)`` — an upper bound on the change
    in expected minimum simple regret between consecutive BO iterations —
    drops below the threshold.

    Three threshold modes:

    - ``"adaptive"`` (default): theoretically motivated threshold from the
      GP noise and posterior variances (Ishibashi et al., 2023).
    - ``"median"``: heuristic ``rate * median(early values)`` over the
      first ``start_timing`` stopping values. The value sequence is anchored
      to the recorded history: values are defined from ``min_experiments``
      observations onward, regardless of when ``has_converged`` is first
      called. Set ``min_experiments`` to the number of experiments at which
      convergence checking effectively begins (e.g. the size of the initial
      design), so the median window covers the intended early phase.
    - ``"adaptive_median"``: converged as soon as either of the two thresholds
      fires.

    The criterion compares consecutive GP posteriors and derives everything it
    needs from the recorded experiments, so it works across process restarts;
    checks in a freshly (re)started process cost one extra GP fit — several in
    ``"median"`` mode.

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``).
    Single-objective only.

    Reference:
        Ishibashi et al. (2023): "A stopping criterion for Bayesian optimization
        by the gap of expected minimum simple regrets" (AISTATS 2023).

    """

    type: Literal["ExpMinRegretGapCriterion"] = "ExpMinRegretGapCriterion"
    threshold_mode: Literal["adaptive", "median", "adaptive_median"] = Field(
        default="adaptive",
        description="How the stopping threshold is computed: the theoretically "
        'motivated "adaptive" threshold, the heuristic "median" threshold, or '
        '"adaptive_median", converged as soon as either fires.',
    )
    delta: PositiveFloat = Field(
        default=0.1,
        description="Confidence parameter for the GP-UCB beta and the adaptive "
        "threshold.",
    )
    rate: PositiveFloat = Field(
        default=0.1,
        description="Fraction of the median stopping value used as the median "
        "threshold.",
    )
    start_timing: PositiveInt = Field(
        default=10,
        description="Number of early stopping values the median threshold is "
        "computed from.",
    )
    min_experiments: PositiveInt = Field(
        default=5,
        description="Minimum experiments before convergence is checked; also "
        "anchors the median mode's stopping-value sequence, so set it to the "
        "number of experiments at which checking effectively begins.",
    )
    beta_scale: PositiveFloat = Field(
        default=1.0,
        description="Scaling factor for the GP-UCB beta.",
    )
    n_samples_lcb: PositiveInt = Field(
        default=1000,
        description="Random domain points for the min-LCB estimate in kappa.",
    )
    noise_var_override: PositiveFloat | None = Field(
        default=None,
        description="If set, replaces the GP's learned noise variance in the "
        "adaptive threshold; use a small value (e.g. 1e-6) for noise-free "
        "objectives.",
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
