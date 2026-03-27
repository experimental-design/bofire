from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective


# ──────────────────────────────────────────────────────────────────────
# Threshold functions for BO stopping
# ──────────────────────────────────────────────────────────────────────


def compute_threshold_noise(
    noise_variance: Optional[float],
    threshold_factor: float = 1.0,
) -> Optional[float]:
    """Compute a threshold from a noise variance.

    Works for both a user-specified (manual) noise variance and a GP-estimated
    one.  Returns ``None`` when the noise variance is unavailable or
    non-positive.

    Args:
        noise_variance: Observation noise variance.  May be ``None`` when
            the GP estimate is not yet available.
        threshold_factor: Multiplier for the threshold.

    Returns:
        threshold_factor * noise_variance, or ``None``.
    """
    if noise_variance is None or noise_variance <= 0:
        return None
    return threshold_factor * noise_variance


def compute_threshold_range(
    experiments: pd.DataFrame,
    output_key: str,
    threshold_factor: float = 1.0,
) -> Optional[float]:
    """Compute a threshold as a fraction of the observed output range.

    Returns ``None`` when the range cannot be computed (fewer than 2
    observations or zero range).

    Args:
        experiments: Experiments conducted so far.
        output_key: Name of the output column.
        threshold_factor: Multiplier for the range.

    Returns:
        threshold_factor * (max(y) - min(y)), or ``None``.
    """
    y_values = experiments[output_key].dropna()
    if len(y_values) < 2:
        return None
    observed_range = float(y_values.max() - y_values.min())
    if observed_range <= 0:
        return None
    return threshold_factor * observed_range


def compute_threshold_cv(
    experiments: pd.DataFrame,
    output_key: str,
    cv_fold_columns: List[str],
    threshold_factor: float = 1.0,
) -> Optional[float]:
    """Compute a threshold from cross-validation fold variability.

    Uses the corrected standard deviation of the incumbent's per-fold
    scores (C. Nadeau and Y. Bengio, NeurIPS 2003):

        threshold = threshold_factor * sqrt(1/K + 1/(K-1)) * std(fold_scores)

    where K is the number of folds and ``ddof=0`` is used to match the
    paper's 1/K divisor. The incumbent is the row with the minimum value
    of *output_key* (assumes minimisation).

    Returns ``None`` when fold scores contain NaN or have zero variability.

    Args:
        experiments: Experiments conducted so far.
        output_key: Name of the output column (used to find the incumbent).
        cv_fold_columns: Column names containing per-fold CV scores.
        threshold_factor: Multiplier (``decay`` in Makarova et al. 2022).

    Returns:
        The corrected CV threshold, or ``None``.
    """
    y_values = experiments[output_key].dropna()
    if len(y_values) < 1:
        return None
    incumbent_idx = y_values.idxmin()
    fold_scores = (
        experiments.loc[incumbent_idx, cv_fold_columns]
        .values.astype(float)
    )
    if np.any(np.isnan(fold_scores)):
        return None
    k = len(cv_fold_columns)
    correction = np.sqrt(1.0 / k + 1.0 / (k - 1))
    fold_std = float(np.std(fold_scores, ddof=0))
    if fold_std <= 0:
        return None
    return float(threshold_factor * correction * fold_std)





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
            Union[
                NumberOfExperimentsCondition,
                "CombiCondition",
                AlwaysTrueCondition,
                "UCBLCBRegretBoundCondition",
            ]
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
    """Condition that checks the UCB-LCB regret bound from Makarova et al. (2022).

    Returns True (keep using this step) while the regret bound is above the
    threshold, and False (move to next step / stop) when the regret bound is
    small enough.

    The regret bound is computed as:
        regret_bound = min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)

    Where (GP-UCB style):
    - UCB(x) = mu(x) + sqrt(beta) * sigma(x)
    - LCB(x) = mu(x) - sqrt(beta) * sigma(x)

    The threshold (epsilon_BO) is determined by the ``noise_variance`` setting:

    - **GP noise (default)**: ``noise_variance=None``. The noise variance is
      estimated from the fitted GP model's ``likelihood.noise``. The threshold
      is ``threshold_factor * estimated_noise_variance``. This follows the
      original Makarova et al. paper and terminates when the regret bound is
      comparable to the observation noise.

    - **Range-based**: ``noise_variance="range"``. The threshold is set as
      ``threshold_factor * (max(y) - min(y))`` — a fraction of the observed
      output range. This requires no noise estimate and is useful for
      noiseless problems or when the GP noise estimate is unreliable.
      Typical ``threshold_factor`` values are 0.01–0.05 (1–5% of range).

    - **CV noise**: ``noise_variance="cv"``. The threshold is derived from
      cross-validation fold score variability of the incumbent (current best).
      Uses the corrected standard deviation from Nadeau and Bengio (2003):
      ``threshold_factor * sqrt(1/K + 1/(K-1)) * std(fold_scores)``
      where K is the number of folds and ``fold_scores`` are the per-fold
      metrics of the incumbent experiment. Requires ``cv_fold_columns`` to
      specify which columns in the experiments DataFrame contain fold scores.

    - **Manual**: Set ``noise_variance`` to a positive float. The threshold
      becomes ``threshold_factor * noise_variance``, using the user-specified
      noise level. This is appropriate when the observation noise is known.

    The condition terminates (returns False) when:
        regret_bound < epsilon_BO

    Top-q filtering can be enabled via ``topq`` to fit the regret-bound GP
    on only the best fraction of observations. This focuses the GP's
    modelling capacity on the promising region of the search space, yielding
    tighter confidence bounds around the optimum. Only the internal GP used
    for termination checking is affected — the main BO strategy's GP is
    unchanged.

    This condition requires a fitted GP-based strategy (e.g., SoboStrategy).
    The strategy is passed via the ``strategy`` keyword argument from the
    StepwiseStrategy.

    Reference:
        Makarova et al. (2022): "Automatic Termination for Hyperparameter
        Optimization" (AutoML 2022)

    Attributes:
        noise_variance: Controls how the termination threshold is computed.
            - ``None`` (default): use GP estimated noise from ``likelihood.noise``.
            - ``"range"``: use ``threshold_factor * observed_output_range``.
            - ``"cv"``: use corrected CV fold std of the incumbent. Requires
              ``cv_fold_columns``.
            - A positive float: use that value directly as the noise variance.
        threshold_factor: Multiplier for the threshold. Default 1.0.
            For CV mode, this corresponds to the ``decay`` parameter in
            Makarova et al. (2022): threshold_factor=1.0 gives the base
            corrected std.
        cv_fold_columns: Column names in the experiments DataFrame containing
            per-fold CV scores. Required when ``noise_variance="cv"``.
            E.g., ["y_fold_0", "y_fold_1", ..., "y_fold_9"] for 10-fold CV.
        topq: Fraction of best observations to use for the regret-bound GP.
            Between 0 (exclusive) and 1 (inclusive). Default 1.0 (no filtering).
            E.g., ``topq=0.5`` fits the GP on the best 50% of observations.
            Only affects the internal termination GP — the main strategy's GP
            is unchanged. A floor of ``min_topq`` observations is always kept.
        min_topq: Minimum number of observations for top-q filtering.
            Default 20. Ensures the GP always has enough data for reliable
            hyperparameter estimation.
        min_experiments: Minimum number of experiments before checking
            termination.
    """

    type: Literal["UCBLCBRegretBoundCondition"] = "UCBLCBRegretBoundCondition"
    noise_variance: Optional[Union[PositiveFloat, Literal["range", "cv"]]] = None
    threshold_factor: PositiveFloat = 1.0
    cv_fold_columns: Optional[List[str]] = None
    topq: Annotated[float, Field(gt=0, le=1)] = 1.0
    min_topq: PositiveInt = 20
    min_experiments: PositiveInt = 5

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
        """Check if optimization should continue (True) or stop (False).

        Args:
            domain: The optimization domain.
            experiments: Experiments conducted so far.
            **kwargs: Must include ``strategy`` — the fitted BotorchStrategy.

        Returns:
            True if optimization should continue, False if converged.
        """
        strategy = kwargs.get("strategy")

        # Keep going if no strategy or model not ready
        if strategy is None:
            return True
        if not getattr(strategy, "is_fitted", False) or getattr(
            strategy, "model", None
        ) is None:
            return True

        # Keep going if not enough data
        if experiments is None or len(experiments) < self.min_experiments:
            return True

        # Compute regret bound via evaluator (lazy import to avoid heavy deps)
        from bofire.termination.evaluator import UCBLCBRegretEvaluator

        evaluator = UCBLCBRegretEvaluator()

        # Top-q filtering: fit a separate GP on the best fraction of
        # observations.  Only affects the regret-bound computation — the
        # main strategy's GP is unchanged.
        eval_strategy = strategy
        eval_experiments = experiments
        if self.topq < 1.0:
            output_key = domain.outputs.get_keys()[0]
            y_values = experiments[output_key].values
            n = len(y_values)
            topn = max(self.min_topq, int(n * self.topq))
            if topn < n:
                top_indices = np.argsort(y_values)[:topn]
                eval_experiments = experiments.iloc[top_indices].reset_index(
                    drop=True,
                )
                # Fit a clone of the main strategy on the filtered subset.
                # Uses the exact same data model (surrogate, kernel, acqf,
                # etc.) — only the training data differs.
                from bofire.strategies.mapper import map as map_strategy

                try:
                    eval_strategy = map_strategy(strategy._data_model)
                    eval_strategy.tell(eval_experiments)
                except Exception:
                    return True  # GP fitting failed, keep going

        metrics = evaluator.evaluate(
            eval_strategy, eval_experiments, len(experiments),
        )

        if not metrics:
            return True  # Evaluation failed, keep going

        regret_bound = metrics["regret_bound"]

        # Determine threshold (epsilon_BO) via reusable helpers
        output_key = domain.outputs.get_keys()[0]

        if isinstance(self.noise_variance, (int, float)):
            epsilon_bo = compute_threshold_noise(
                self.noise_variance, self.threshold_factor,
            )
        elif self.noise_variance == "range":
            epsilon_bo = compute_threshold_range(
                experiments, output_key, self.threshold_factor,
            )
        elif self.noise_variance == "cv":
            epsilon_bo = compute_threshold_cv(
                experiments, output_key, self.cv_fold_columns,
                self.threshold_factor,
            )
        else:
            epsilon_bo = compute_threshold_noise(
                metrics.get("estimated_noise_variance"),
                self.threshold_factor,
            )

        if epsilon_bo is None:
            return True  # Threshold could not be computed, keep going

        # Return True (keep going) if regret bound is still large
        return regret_bound >= epsilon_bo


AnyCondition = Union[
    NumberOfExperimentsCondition,
    CombiCondition,
    AlwaysTrueCondition,
    FeasibleExperimentCondition,
    UCBLCBRegretBoundCondition,
]
