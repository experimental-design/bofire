from typing import Literal, Optional

from pydantic import PositiveFloat, PositiveInt

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class ExpMinRegretGapCriterion(ConvergenceCriterion):
    """Expected minimum regret gap criterion (Ishibashi et al. 2023).

    The optimization is considered converged once the stopping value
    ``delta_f + ei_diff + kappa * sqrt(KL / 2)`` — an upper bound on the change
    in expected minimum simple regret between consecutive BO iterations —
    drops below the threshold.

    Two threshold modes:

    - ``"adaptive"`` (default): theoretically motivated threshold from the
      GP noise and posterior variances (Ishibashi et al., 2023).
    - ``"median"``: heuristic ``rate * median(early values)`` over the
      first ``start_timing`` stopping values. The value sequence is anchored
      to the recorded history: values are defined from ``min_experiments``
      observations onward, regardless of when ``has_converged`` is first
      called. Set ``min_experiments`` to the number of experiments at which
      convergence checking effectively begins (e.g. the size of the initial
      design), so the median window covers the intended early phase.

    The criterion compares consecutive GP posteriors. In a running BO loop
    (one new experiment per check) the previous posterior is reused from a
    per-strategy cache, so consecutive checks need no extra GP fits. On a cold
    start (fresh or deserialized strategy) the previous-iteration model is
    reconstructed purely from the recorded history by refitting on all but the
    last experiment; the ``"median"`` mode additionally replays the early
    stopping values from prefixes of the history, which makes cold starts in
    that mode considerably more expensive. Results never depend on the cache —
    it is invalidated whenever it cannot seamlessly continue, falling back to
    the pure reconstruction.

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``).
    Single-objective only.

    Reference:
        Ishibashi et al. (2023): "A stopping criterion for Bayesian optimization
        by the gap of expected minimum simple regrets" (AISTATS 2023).

    Attributes:
        threshold_mode: ``"adaptive"`` or ``"median"``.
        delta: Confidence parameter for beta and the adaptive threshold.
        rate: Fraction of the median stopping value used as threshold in
            ``"median"`` mode.
        start_timing: Stopping values collected before the median threshold
            can be computed / the criterion can trigger.
        min_experiments: Minimum experiments before checking.
        beta_scale: Scaling factor for the GP-UCB beta parameter.
        n_samples_lcb: Random samples for the min-LCB estimate in kappa.
        noise_var_override: If set, replaces the GP's learned noise variance
            when computing the adaptive threshold.  Use a small value (e.g.
            ``1e-6``) for exact (noise-free) objectives, where the GP can
            otherwise over-estimate noise early and trigger a premature stop.
    """

    type: Literal["ExpMinRegretGapCriterion"] = "ExpMinRegretGapCriterion"
    threshold_mode: Literal["adaptive", "median"] = "adaptive"
    delta: PositiveFloat = 0.1
    rate: PositiveFloat = 0.1
    start_timing: PositiveInt = 10
    min_experiments: PositiveInt = 5
    beta_scale: PositiveFloat = 1.0
    n_samples_lcb: PositiveInt = 1000
    noise_var_override: Optional[PositiveFloat] = None

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
