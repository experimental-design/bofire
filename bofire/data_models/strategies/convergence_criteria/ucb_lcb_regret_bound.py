from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)


class UCBLCBRegretBoundCriterion(ConvergenceCriterion):
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

    Attributes:
        noise_variance: Noise variance source (see description).
        threshold_factor: Multiplier for the threshold (``decay`` in
            Makarova et al. 2022 for the CV mode).
        cv_fold_columns: Column names with per-fold CV scores; required
            when ``noise_variance="cv"``.
        topq: Fraction of best observations used for the internal
            regret-bound GP. Default ``0.5`` — Makarova et al. (2022) found
            fitting the bound on the best ~50 % of observations works best.
            Set to ``1.0`` to disable filtering and use all observations. The
            main strategy's GP is unaffected. Only engages once more than
            ``min_topq`` observations are available.
        min_topq: Minimum observations kept under top-q filtering.
        min_experiments: Minimum experiments before convergence is checked.
        delta: Confidence parameter for the GP-UCB beta formula. Default ``0.1``.
        beta_scale: Scaling factor for the GP-UCB beta. Default ``0.2``
            (Makarova et al.).
        n_samples_lcb: Random domain points for the min-LCB estimate when
            ``lcb_method="sample"``. Default ``2000``.
        batch_size: If set, chunk GP posterior evaluation into batches of this
            size during sampling to bound memory.  ``None`` (default) evaluates
            all points in a single posterior call.
        lcb_method: How the domain-wide minimum LCB is found — ``"sample"``
            (default) draws random points; ``"optimize"`` uses the acquisition
            optimizer.
        fallback_noise_variance: Noise variance used when it cannot be read
            from the GP likelihood. Default ``1e-4``.
    """

    type: Literal["UCBLCBRegretBoundCriterion"] = "UCBLCBRegretBoundCriterion"
    noise_variance: Optional[Union[PositiveFloat, Literal["cv"]]] = None
    threshold_factor: PositiveFloat = 1.0
    cv_fold_columns: Optional[List[str]] = None
    topq: Annotated[float, Field(gt=0, le=1)] = 0.5
    min_topq: PositiveInt = 20
    min_experiments: PositiveInt = 5
    delta: PositiveFloat = 0.1
    beta_scale: PositiveFloat = 0.2
    n_samples_lcb: PositiveInt = 2000
    batch_size: Optional[PositiveInt] = None
    lcb_method: Literal["sample", "optimize"] = "sample"
    fallback_noise_variance: PositiveFloat = 1e-4

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
