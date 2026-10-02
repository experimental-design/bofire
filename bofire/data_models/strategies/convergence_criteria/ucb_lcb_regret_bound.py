from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class UcbLcbRegretBoundCriterion(ConvergenceCriterion):
    """Convergence based on the UCB-LCB regret bound from Makarova et al. (2022).

    The optimization is considered converged once the regret bound
    ``min_x_evaluated UCB(x) - min_x_domain LCB(x)`` drops below the threshold
    ``epsilon_BO``, using GP-UCB style bounds ``mu(x) ± sqrt(beta) * sigma(x)``.

    The threshold ``epsilon_BO`` depends on ``noise_variance``:

    - ``None`` (default): GP-estimated noise ``likelihood.noise``.
    - ``"cv"``: corrected CV-fold std of the incumbent
      (Nadeau and Bengio, 2003). Requires ``cv_fold_columns``.
    - positive float: used directly as the noise variance.

    In all cases the threshold is ``threshold_factor * <noise_variance>``.

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``); the evaluator
    reads the fitted model directly from the strategy. Single-objective only.

    Reference:
        Makarova et al. (2022): "Automatic Termination for Hyperparameter
        Optimization" (AutoML 2022).

    """

    type: Literal["UcbLcbRegretBoundCriterion"] = "UcbLcbRegretBoundCriterion"
    noise_variance: PositiveFloat | Literal["cv"] | None = Field(
        default=None,
        description="Source of the noise variance the threshold is built from: "
        "a positive float is used directly, None uses the GP-estimated noise, "
        'and "cv" uses the corrected CV-fold std of the incumbent.',
    )
    threshold_factor: PositiveFloat = Field(
        default=1.0,
        description="Multiplier on the noise variance; the threshold is "
        "`threshold_factor * noise_variance`.",
    )
    cv_fold_columns: list[str] | None = Field(
        default=None,
        description="Experiments columns holding per-fold CV scores; required "
        'when `noise_variance="cv"`.',
    )
    topq: Annotated[float, Field(gt=0, le=1)] = Field(
        default=0.5,
        description="Fraction of the best observations the regret-bound GP is "
        "refit on (Makarova et al. recommend ~0.5); 1.0 disables the filtering.",
    )
    min_topq: PositiveInt = Field(
        default=20,
        description="Minimum observations kept under top-q filtering.",
    )
    min_experiments: PositiveInt = Field(
        default=5,
        description="Minimum experiments before convergence is checked.",
    )
    delta: PositiveFloat = Field(
        default=0.1,
        description="Confidence parameter of the GP-UCB beta formula.",
    )
    beta_scale: PositiveFloat = Field(
        default=0.2,
        description="Scaling factor for the GP-UCB beta.",
    )
    n_samples_lcb: PositiveInt = Field(
        default=2000,
        description="Random domain points for the min-LCB estimate when "
        '`lcb_method="sample"`.',
    )
    batch_size: PositiveInt | None = Field(
        default=None,
        description="If set, chunk GP posterior evaluation into batches of "
        "this size to bound memory.",
    )
    lcb_method: Literal["sample", "optimize"] = Field(
        default="sample",
        description="How the domain-wide minimum LCB is found: random "
        "sampling or the acquisition optimizer.",
    )
    fallback_noise_variance: PositiveFloat = Field(
        default=1e-4,
        description="Noise variance used when it cannot be read from the GP "
        "likelihood.",
    )

    @model_validator(mode="after")
    def validate_cv_fold_columns(self):
        if self.noise_variance == "cv":
            if self.cv_fold_columns is None or len(self.cv_fold_columns) < 2:
                raise ValueError(
                    "cv_fold_columns must be a list of at least 2 column names "
                    'when noise_variance="cv".',
                )
        return self

    @classmethod
    def is_applicable_to_multiobjective(cls) -> bool:
        return False

    @classmethod
    def is_applicable_to_singleobjective(cls) -> bool:
        return True

    @classmethod
    def is_applicable_to_objective_free(cls) -> bool:
        return False
