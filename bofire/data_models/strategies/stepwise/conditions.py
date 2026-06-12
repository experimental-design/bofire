from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Union

import pandas as pd
from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.unions import tagged_union


class EvaluateableCondition:
    @abstractmethod
    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        pass


class Condition(BaseModel):
    type: Any


class SingleCondition(BaseModel):
    type: Any


class FeasibleExperimentCondition(SingleCondition, EvaluateableCondition):
    """Condition to check if a certain number of feasible experiments are available.

    For this purpose, the condition checks if there are any kind of ConstrainedObjective's
    in the domain. If, yes it checks if there is a certain number of feasible experiments.
    The condition is fulfilled if the number of feasible experiments is smaller than
    the number of required feasible experiments. It is not fulfilled when there are no
    ConstrainedObjective's in the domain.
    This condition can be used in scenarios where there is a large amount of output constraints
    and one wants to make sure that they are fulfilled before optimizing the actual objective(s).
    To do this, it is best to combine this condition with the SoboStrategy and qLogPF
    as acquisition function.

    Attributes:
        n_required_feasible_experiments: Number of required feasible experiments.
        threshold: Threshold for the feasibility calculation. Default is 0.9.
    """

    type: Literal["FeasibleExperimentCondition"] = "FeasibleExperimentCondition"
    n_required_feasible_experiments: PositiveInt = 1
    threshold: Annotated[float, Field(ge=0, le=1)] = 0.9

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        constrained_outputs = domain.outputs.get_by_objective(ConstrainedObjective)
        if len(constrained_outputs) == 0:
            return False

        if experiments is None:
            return True

        valid_experiments = (
            constrained_outputs.preprocess_experiments_all_valid_outputs(experiments)
        )

        valid_experiments = valid_experiments[domain.is_fulfilled(valid_experiments)]

        feasibilities = pd.concat(
            [
                feat(
                    valid_experiments[feat.key],
                    valid_experiments[feat.key],
                )
                for feat in constrained_outputs
            ],
            axis=1,
        ).product(axis=1)

        return bool(
            feasibilities[feasibilities >= self.threshold].sum()
            < self.n_required_feasible_experiments
        )


class NumberOfExperimentsCondition(SingleCondition, EvaluateableCondition):
    type: Literal["NumberOfExperimentsCondition"] = "NumberOfExperimentsCondition"
    n_experiments: Annotated[int, Field(ge=1)]

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        if experiments is None:
            n_experiments = 0
        else:
            n_experiments = len(
                domain.outputs.preprocess_experiments_all_valid_outputs(experiments),
            )
        return n_experiments < self.n_experiments


class AlwaysTrueCondition(SingleCondition, EvaluateableCondition):
    type: Literal["AlwaysTrueCondition"] = "AlwaysTrueCondition"

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        return True


class CombiCondition(Condition, EvaluateableCondition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        List[
            tagged_union(
                NumberOfExperimentsCondition,
                "CombiCondition",
                AlwaysTrueCondition,
                "UCBLCBRegretBoundCondition",
                "ExpMinRegretGapCondition",
                "LogEIPCCondition",
                "ProbabilisticRegretBoundCondition",
            )
        ],
        Field(min_length=2),
    ]
    n_required_conditions: Annotated[int, Field(ge=0)]

    @field_validator("n_required_conditions")
    @classmethod
    def validate_n_required_conditions(cls, v, info):
        if v > len(info.data["conditions"]):
            raise ValueError(
                "Number of required conditions larger than number of conditions.",
            )
        return v

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        n_matched_conditions = 0
        for c in self.conditions:
            if c.evaluate(domain, experiments, **kwargs):
                n_matched_conditions += 1
        if n_matched_conditions >= self.n_required_conditions:
            return True
        return False


class UCBLCBRegretBoundCondition(SingleCondition, EvaluateableCondition):
    """Condition based on the UCB-LCB regret bound from Makarova et al. (2022).

    Returns ``True`` (keep going) while the regret bound
    ``min_x_evaluated UCB(x) - min_x_domain LCB(x)`` exceeds the threshold
    ``epsilon_BO``, using GP-UCB style bounds
    ``mu(x) ± sqrt(beta) * sigma(x)``.

    The threshold ``epsilon_BO`` depends on ``noise_variance``:

    - ``None`` (default): GP-estimated noise ``likelihood.noise``.
    - ``"cv"``: corrected CV-fold std of the incumbent
      (Nadeau and Bengio, 2003). Requires ``cv_fold_columns``.
    - positive float: used directly as the noise variance.

    In all cases the threshold is ``threshold_factor * <noise_variance>``.

    Requires a fitted GP-based strategy (e.g. ``SoboStrategy``) passed via
    the ``strategy`` kwarg from the ``StepwiseStrategy``.

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
        min_experiments: Minimum experiments before termination is checked.
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

    type: Literal["UCBLCBRegretBoundCondition"] = "UCBLCBRegretBoundCondition"
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

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        """Check if optimization should continue (``True``) or stop (``False``).

        Args:
            domain: The optimization domain.
            experiments: Experiments conducted so far.
            **kwargs: Must include ``strategy`` — the fitted ``BotorchStrategy``.

        Returns:
            ``True`` if optimization should continue, ``False`` if the
            regret bound has dropped below ``epsilon_BO``.
        """
        strategy = kwargs.get("strategy")

        if strategy is None:
            return True
        if (
            not getattr(strategy, "is_fitted", False)
            or getattr(strategy, "model", None) is None
        ):
            return True

        if experiments is None or len(experiments) < self.min_experiments:
            return True

        from bofire.strategies.stepwise.termination.ucb_lcb import UCBLCBRegretEvaluator

        evaluator = UCBLCBRegretEvaluator(
            delta=self.delta,
            beta_scale=self.beta_scale,
            fallback_noise_variance=self.fallback_noise_variance,
            n_samples_lcb=self.n_samples_lcb,
            batch_size=self.batch_size,
            lcb_method=self.lcb_method,
            topq=self.topq,
            min_topq=self.min_topq,
        )

        # Objective direction (+1 maximise / -1 minimise, BoFire convention);
        # the regret bound does not apply to other objectives, so keep
        # optimizing.  Negate to the minimisation frame used for the CV
        # threshold below.
        direction = evaluator._objective_sign(strategy)
        if direction is None:
            return True
        sign = -direction  # +1 minimise / -1 maximise

        metrics = evaluator.evaluate(strategy, experiments, len(experiments))

        if not metrics:
            return True

        regret_bound = metrics["regret_bound"]

        from bofire.strategies.stepwise.termination.utils import (
            compute_threshold_cv,
            compute_threshold_noise,
        )

        output_key = domain.outputs.get_keys()[0]

        if isinstance(self.noise_variance, (int, float)):
            epsilon_bo = compute_threshold_noise(
                self.noise_variance,
                self.threshold_factor,
            )
        elif self.noise_variance == "cv":
            epsilon_bo = compute_threshold_cv(
                experiments,
                output_key,
                self.cv_fold_columns,
                self.threshold_factor,
                sign=sign,
            )
        else:
            epsilon_bo = compute_threshold_noise(
                metrics.get("estimated_noise_variance"),
                self.threshold_factor,
            )

        if epsilon_bo is None:
            return True

        return regret_bound >= epsilon_bo


class ExpMinRegretGapCondition(SingleCondition, EvaluateableCondition):
    """Expected minimum regret gap stopping criterion (Ishibashi et al. 2023).

    Returns ``True`` (keep optimising) while the stopping value
    ``delta_f + ei_diff + kappa * sqrt(KL / 2)`` exceeds the threshold,
    and ``False`` (stop) otherwise.

    Two threshold modes:

    - ``"adaptive"`` (default): theoretically motivated threshold from the
      GP noise and posterior variances (Ishibashi et al., 2023).
    - ``"median"``: heuristic ``rate * median(early values)`` over the
      first ``start_timing`` stopping values.

    Stateful: keeps the previous-iteration GP model to compare consecutive
    posteriors. Always returns ``True`` before ``min_experiments`` is
    reached. The state lives in a private attribute and is not serialized;
    after deserialization the criterion silently re-primes on the next call.

    Reference:
        Ishibashi et al. (2023): "A stopping criterion for Bayesian optimization
        by the gap of expected minimum simple regrets" (AISTATS 2023).

    Attributes:
        threshold_mode: ``"adaptive"`` or ``"median"``.
        delta: Confidence parameter for beta and the adaptive threshold.
        rate: Fraction of the median stopping value used as threshold in
            ``"median"`` mode.
        start_timing: Stopping values collected before the median threshold
            can be computed / the condition can trigger.
        min_experiments: Minimum experiments before checking.
        beta_scale: Scaling factor for the GP-UCB beta parameter.
        n_samples_lcb: Random samples for the min-LCB estimate in kappa.
        noise_var_override: If set, replaces the GP's learned noise variance
            when computing the adaptive threshold.  Use a small value (e.g.
            ``1e-6``) for exact (noise-free) objectives, where the GP can
            otherwise over-estimate noise early and trigger a premature stop.
    """

    type: Literal["ExpMinRegretGapCondition"] = "ExpMinRegretGapCondition"
    threshold_mode: Literal["adaptive", "median"] = "adaptive"
    delta: PositiveFloat = 0.1
    rate: PositiveFloat = 0.1
    start_timing: PositiveInt = 10
    min_experiments: PositiveInt = 5
    beta_scale: PositiveFloat = 1.0
    n_samples_lcb: PositiveInt = 1000
    noise_var_override: Optional[PositiveFloat] = None

    _evaluator: Any = PrivateAttr(default=None)

    def _get_evaluator(self):
        if self._evaluator is None:
            from bofire.strategies.stepwise.termination.exp_min_regret_gap import (
                ExpMinRegretGapEvaluator,
            )

            self._evaluator = ExpMinRegretGapEvaluator(
                delta=self.delta,
                rate=self.rate,
                start_timing=self.start_timing,
                beta_scale=self.beta_scale,
                n_samples_lcb=self.n_samples_lcb,
                noise_var_override=self.noise_var_override,
                # Compute both thresholds; ``evaluate`` below picks the one
                # matching this condition's ``threshold_mode``.
                threshold_mode="adaptive_median",
            )
        return self._evaluator

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        """Check if optimization should continue (``True``) or stop (``False``).

        Args:
            domain: The optimization domain.
            experiments: Experiments conducted so far.
            **kwargs: Must include ``strategy`` — the fitted ``BotorchStrategy``.

        Returns:
            ``True`` if optimization should continue, ``False`` when the
            stopping value drops below the selected threshold.
        """
        strategy = kwargs.get("strategy")

        if strategy is None:
            return True
        if (
            not getattr(strategy, "is_fitted", False)
            or getattr(strategy, "model", None) is None
        ):
            return True

        if experiments is None or len(experiments) < self.min_experiments:
            return True

        evaluator = self._get_evaluator()
        metrics = evaluator.evaluate(strategy, experiments, len(experiments))

        if not metrics:
            return True

        stopping_value = metrics["stopping_value"]

        if self.threshold_mode == "adaptive":
            threshold = metrics.get("threshold_adaptive")
        else:
            threshold = metrics.get("threshold_median")

        if threshold is None:
            return True

        return bool(stopping_value >= threshold)


class LogEIPCCondition(SingleCondition, EvaluateableCondition):
    """Cost-aware stopping criterion (Xie et al., 2025).

    Stops (returns ``False``) when the maximum log expected-improvement-per-cost
    over the domain drops to zero or below — i.e. no unevaluated point's
    expected improvement is worth its evaluation cost:

        stop when  max_x [ LogEI(x) - alpha * log(c(x)) - log(lambda_cost) ] <= 0

    Ideal for chemical experiments where reagent, time, or equipment costs
    matter. The ``cost_column`` attribute lets you record the actual cost of
    each experiment and use the running mean as the cost estimate.

    Attributes:
        lambda_cost: Exchange rate between cost and improvement. Higher values
            favour earlier stopping (require higher improvement-to-cost ratio
            to continue). Default ``1.0``.
        cost_column: Name of the column in the experiments DataFrame that
            records the cost of each experiment. When set, the mean of past
            costs is used as the cost estimate. Takes priority over
            ``cost_value``.
        cost_value: Fixed cost per experiment used when ``cost_column`` is not
            provided. Default ``1.0``.
        alpha: Exponent applied to the cost in the LogEIPC formula. ``1.0``
            (default) matches the paper's primary formulation.
        min_experiments: Minimum experiments before the condition is checked.
            Default ``5``.
        n_samples: Random domain samples used to approximate the max LogEIPC.
            Default ``2000``.
        search_method: How to find the max LogEIPC — ``"sample"`` uses random
            grid search (default); ``"optimize"`` uses gradient-based search.
        cost_model: How cost is estimated — ``"mean"`` uses the running mean
            of past costs (default); ``"gp"`` fits a GP to predict cost.

    Reference:
        Xie et al. (2025): "Cost-Aware Stopping for Bayesian Optimization"
        (arXiv:2507.12453).
    """

    type: Literal["LogEIPCCondition"] = "LogEIPCCondition"
    lambda_cost: PositiveFloat = 1.0
    cost_column: Optional[str] = None
    cost_value: PositiveFloat = 1.0
    alpha: PositiveFloat = 1.0
    min_experiments: PositiveInt = 5
    n_samples: PositiveInt = 2000
    search_method: Literal["sample", "optimize"] = "sample"
    cost_model: Literal["mean", "gp"] = "mean"

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        """Check if optimization should continue (``True``) or stop (``False``).

        Args:
            domain: The optimization domain.
            experiments: Experiments conducted so far.
            **kwargs: Must include ``strategy`` — the fitted ``BotorchStrategy``.

        Returns:
            ``True`` if optimization should continue, ``False`` when
            ``max_log_eipc <= 0``.
        """
        strategy = kwargs.get("strategy")

        if strategy is None:
            return True
        if (
            not getattr(strategy, "is_fitted", False)
            or getattr(strategy, "model", None) is None
        ):
            return True

        if experiments is None or len(experiments) < self.min_experiments:
            return True

        from bofire.strategies.stepwise.termination.log_eipc import LogEIPCEvaluator

        evaluator = LogEIPCEvaluator(
            lambda_cost=self.lambda_cost,
            cost_column=self.cost_column,
            cost_value=self.cost_value,
            alpha=self.alpha,
            n_samples=self.n_samples,
            search_method=self.search_method,
            cost_model=self.cost_model,
        )
        metrics = evaluator.evaluate(strategy, experiments, len(experiments))

        if not metrics:
            return True

        return bool(metrics["max_log_eipc"] > 0)


class ProbabilisticRegretBoundCondition(SingleCondition, EvaluateableCondition):
    """Stopping condition based on probabilistic regret bounds (Wilson, 2024).

    Returns ``True`` (keep optimizing) while the estimated probability that
    the incumbent's regret exceeds ε is above the model-risk threshold.
    Returns ``False`` (stop) once the Clopper-Pearson sequential hypothesis
    test certifies the criterion has been met.

    The two risk parameters ``delta_mod`` and ``delta_est`` correspond directly
    to the paper's δ_mod (model risk) and δ_est (estimation error from the
    Monte Carlo test).  Stopping triggers when ``P̂(regret > ε) ≤ δ_mod``,
    certified by a Clopper-Pearson test at level δ_est.

    **To reproduce the paper's main experiments** use ``delta_mod = delta_est
    = 0.025`` (total risk 5 %, 95 % guarantee).

    Requires a fitted GP-based strategy passed via the ``strategy`` kwarg
    from ``StepwiseStrategy``.  Single-output only.

    How ε is determined (in order of priority):

    1. ``epsilon`` — absolute threshold in original Y units, if explicitly set.
    2. ``epsilon_relative × (y_max − y_min)`` — relative threshold (default
       ``1 %`` of the observed range when ``epsilon`` is not set).

    Args:
        epsilon: Absolute simple regret threshold in Y units.  If ``None``
            (default), computed from ``epsilon_relative``.
        epsilon_relative: Fractional ε relative to the observed Y range.
            Default ``0.01`` (1 %).  Ignored when ``epsilon`` is set.
        delta_mod: Model-risk tolerance δ_mod.  Stopping triggers when the
            estimated probability that regret exceeds ε falls below this
            value.  Default ``0.05``.
        delta_est: Estimation-risk tolerance δ_est for the sequential
            Clopper-Pearson test.  Default ``0.05``.
        enforce_convergence: Only stop when the CP CI conclusively excludes
            δ_mod (default ``True``).  Set to ``False`` to use the raw MC
            estimate.
        n_samples_max: Maximum GP path samples per BO step.  Default ``1024``.
        initial_batch: Initial cumulative sample target for the Clopper-Pearson
            level test.  Default ``16``.
        batch_growth: Geometric growth factor for the cumulative sample
            schedule (must be ``> 1``).  Default ``1.5``.
        min_experiments: Minimum experiments before the condition is checked.
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

    type: Literal["ProbabilisticRegretBoundCondition"] = (
        "ProbabilisticRegretBoundCondition"
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

    _evaluator: Any = PrivateAttr(default=None)

    def _get_evaluator(self):
        if self._evaluator is None:
            from bofire.strategies.stepwise.termination.probabilistic_regret_bound import (
                ProbabilisticRegretBoundEvaluator,
            )

            self._evaluator = ProbabilisticRegretBoundEvaluator(
                epsilon=self.epsilon,
                epsilon_relative=self.epsilon_relative,
                delta_mod=self.delta_mod,
                delta_est=self.delta_est,
                enforce_convergence=self.enforce_convergence,
                n_samples_max=self.n_samples_max,
                initial_batch=self.initial_batch,
                batch_growth=self.batch_growth,
                n_starts=self.n_starts,
                n_random=self.n_random,
                n_test_points=self.n_test_points,
                optim_method=self.optim_method,
                optim_maxiter=self.optim_maxiter,
                optim_ftol=self.optim_ftol,
            )
        return self._evaluator

    def evaluate(
        self, domain: Domain, experiments: Optional[pd.DataFrame], **kwargs
    ) -> bool:
        """Check if optimisation should continue (``True``) or stop (``False``).

        Args:
            domain: The optimisation domain.
            experiments: Experiments conducted so far.
            **kwargs: Must include ``strategy`` — the fitted ``BotorchStrategy``.

        Returns:
            ``True`` if optimisation should continue, ``False`` when the
            probabilistic regret bound criterion is satisfied.
        """
        strategy = kwargs.get("strategy")

        if strategy is None:
            return True
        if (
            not getattr(strategy, "is_fitted", False)
            or getattr(strategy, "model", None) is None
        ):
            return True
        if experiments is None or len(experiments) < self.min_experiments:
            return True

        evaluator = self._get_evaluator()
        metrics = evaluator.evaluate(strategy, experiments, len(experiments))

        if not metrics:
            return True

        return not metrics["criterion_satisfied"]


AnyCondition = tagged_union(
    NumberOfExperimentsCondition,
    CombiCondition,
    AlwaysTrueCondition,
    FeasibleExperimentCondition,
    UCBLCBRegretBoundCondition,
    ExpMinRegretGapCondition,
    LogEIPCCondition,
    ProbabilisticRegretBoundCondition,
)
