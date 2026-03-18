"""Termination evaluators that compute metrics for termination conditions.

These evaluators work with BotorchStrategy instances to compute the various
metrics used by termination conditions (e.g., UCB-LCB regret bound).

Used internally by ``UCBLCBRegretBoundCondition`` in the StepwiseStrategy
conditions system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model

from bofire.utils.torch_tools import tkwargs


class TerminationEvaluator(ABC):
    """Base class for termination evaluators.

    Termination evaluators compute metrics from a BO strategy that can be
    used by termination conditions to decide whether to stop optimization.
    """

    @abstractmethod
    def evaluate(
        self,
        strategy,  # BotorchStrategy
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Evaluate termination metrics.

        Args:
            strategy: The BotorchStrategy being used for optimization.
            experiments: All experiments conducted so far.
            iteration: Current iteration number.

        Returns:
            Dictionary of metric names to values (floats, lists, etc.).
        """
        pass


class _NegLowerConfidenceBound(AnalyticAcquisitionFunction):
    """Negative LCB acquisition function for minimizing LCB via BoTorch optimizers.

    Since BoTorch optimizers maximize acquisition functions, we negate LCB to
    find its minimum: argmin LCB(x) = argmax -LCB(x).

    LCB(x) = mu(x) - sqrt(beta) * sigma(x)
    """

    def __init__(self, model: Model, sqrt_beta: float):
        super().__init__(model=model)
        self.sqrt_beta = sqrt_beta

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: batch x q x d, for q=1 analytic case
        posterior = self.model.posterior(X.squeeze(-2))
        mean = posterior.mean.squeeze(-1)  # batch
        std = posterior.variance.squeeze(-1).sqrt()
        lcb = mean - self.sqrt_beta * std
        return -lcb  # negate so maximize -> minimize LCB


class UCBLCBRegretEvaluator(TerminationEvaluator):
    """Evaluator for UCB-LCB regret bound from Makarova et al. (2022).

    Computes the regret bound as:
        regret_bound = min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)

    Where (GP-UCB style, Srinivas et al. 2010):
    - UCB(x) = mu(x) + sqrt(beta_t) * sigma(x)
    - LCB(x) = mu(x) - sqrt(beta_t) * sigma(x)
    - beta_t = beta_scale * 2 * log(d * t^2 * pi^2 / (6 * delta))

    The beta parameter grows logarithmically with the number of observations
    (t) and dimensionality (d). The scaling factor ``beta_scale`` defaults to
    0.2, following the GP-UCB paper's experiments section, but is configurable
    via the class attribute.

    The ``lcb_method`` attribute controls how min LCB over the domain is found:

    - ``"sample"`` (default): Evaluate LCB at a random sample of candidates
      (``n_samples_lcb`` points) merged with observed points, processed in
      batches of ``batch_size``. Fast and memory-efficient. Matches the
      reference implementation at github.com/amazon-science/bo-early-stopping.
    - ``"optimize"``: Use BoTorch's acquisition function optimizer to find the
      global minimum of LCB. More principled but slower, as it runs multi-start
      L-BFGS-B optimization.

    This provides an upper bound on the simple regret of the current best
    recommendation. When this bound becomes small (relative to noise), further
    optimization is unlikely to yield significant improvement.

    Note:
        Currently only supports single-output minimization problems.

    Reference:
        Makarova et al. (2022): "Automatic Termination for Hyperparameter
        Optimization" (AutoML 2022)
        Srinivas et al. (2010): "Gaussian Process Optimization in the Bandit
        Setting: No Regret and Experimental Design" (ICML 2010)
    """

    fallback_noise_variance: float = 1e-4
    delta: float = 0.1
    beta_scale: float = 0.2
    beta_log_multiplier: float = 2.0
    beta_log_denominator: float = 6.0
    beta_min: float = 0.01
    n_samples_lcb: int = 2000
    batch_size: int = 512
    lcb_method: Literal["sample", "optimize"] = "sample"

    def _compute_beta(self, dimensionality: int, num_observed: int) -> float:
        """Compute beta for UCB/LCB using the GP-UCB formula from Srinivas et al. (2010).

        Beta based on "Gaussian Process Optimization in the Bandit Setting:
        No Regret and Experimental Design" (Srinivas et al., 2010).

        beta_t = beta_scale * 2 * log(d * t^2 * pi^2 / (6 * delta))

        where d = dimensionality, t = num_observed, delta = confidence parameter.
        All constants (beta_scale, beta_log_multiplier, etc.) are configurable
        via class attributes.

        Args:
            dimensionality: Number of input dimensions.
            num_observed: Number of observations so far.

        Returns:
            The beta value for UCB/LCB bounds.
        """
        beta = (
            self.beta_scale
            * self.beta_log_multiplier
            * np.log(
                dimensionality
                * num_observed**2
                * np.pi**2
                / (self.beta_log_denominator * self.delta)
            )
        )
        return max(beta, self.beta_min)  # ensure positive

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Compute the UCB-LCB regret bound.

        Computes min UCB over evaluated points directly. For min LCB over the
        domain, uses either random sampling or BoTorch acquisition optimization
        depending on ``self.lcb_method``.

        Beta is computed using the GP-UCB formula from Srinivas et al. (2010),
        scaled by ``beta_scale`` (default 0.2).

        Currently only supports single-output GP models. Returns an empty dict
        for multi-output models or non-fitted strategies.

        Args:
            strategy: A fitted BotorchStrategy.
            experiments: Current experiments.
            iteration: Current iteration.

        Returns:
            Dictionary with 'regret_bound', 'min_ucb_evaluated', 'min_lcb_domain',
            'estimated_noise_variance', and 'beta'. Returns empty dict if the
            strategy is not fitted, has too few experiments, or uses a
            multi-output model.
        """
        if not strategy.is_fitted or strategy.model is None:
            return {}

        if len(experiments) < 2:
            return {}

        # Check for single-output model only
        if strategy.model.num_outputs != 1:
            return {}

        # Compute beta using GP-UCB formula (Srinivas et al. 2010)
        input_keys = strategy.domain.inputs.get_keys()
        dimensionality = len(input_keys)
        num_observed = len(experiments)
        beta = self._compute_beta(dimensionality, num_observed)
        sqrt_beta = np.sqrt(beta)

        # --- min UCB over evaluated points ---
        min_ucb_evaluated, X_evaluated = self._min_ucb_evaluated(
            strategy, experiments, sqrt_beta,
        )

        # --- min LCB over domain ---
        if self.lcb_method == "optimize":
            min_lcb_domain = self._min_lcb_optimize(
                strategy, experiments, sqrt_beta
            )
        else:
            min_lcb_domain = self._min_lcb_sample(
                strategy, X_evaluated, sqrt_beta
            )

        # Regret bound = min_UCB(evaluated) - min_LCB(domain)
        regret_bound = max(0.0, min_ucb_evaluated - min_lcb_domain)

        # Estimate noise variance from GP likelihood
        try:
            estimated_noise_var = strategy.model.likelihood.noise.item()
        except Exception:
            # Some model types may not expose noise via likelihood.
            estimated_noise_var = self.fallback_noise_variance

        return {
            "regret_bound": regret_bound,
            "min_ucb_evaluated": min_ucb_evaluated,
            "min_lcb_domain": min_lcb_domain,
            "estimated_noise_variance": estimated_noise_var,
            "beta": beta,
        }

    def _min_ucb_evaluated(
        self,
        strategy,
        experiments: pd.DataFrame,
        sqrt_beta: float,
    ) -> tuple:
        """Compute min UCB over evaluated (observed) points.

        Evaluates UCB = mu(x) + sqrt(beta) * sigma(x) at every observed
        input point and returns the minimum.

        Also returns the transformed input tensor so it can be reused by
        ``_min_lcb_sample`` (which merges random candidates with evaluated
        points).

        Args:
            strategy: A fitted BotorchStrategy.
            experiments: Current experiments DataFrame.
            sqrt_beta: Square root of beta for UCB computation.

        Returns:
            A tuple ``(min_ucb, X_evaluated)`` where *min_ucb* is a float
            and *X_evaluated* is the transformed input tensor of shape
            ``(N, d_transformed)``.
        """
        input_keys = strategy.domain.inputs.get_keys()
        evaluated_inputs = experiments[input_keys]
        transformed_evaluated = strategy.domain.inputs.transform(
            evaluated_inputs,
            strategy.input_preprocessing_specs,
        )
        X_evaluated = torch.from_numpy(transformed_evaluated.values).to(**tkwargs)

        with torch.no_grad():
            posterior = strategy.model.posterior(X_evaluated)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.squeeze(-1).sqrt()

        ucb = mean + sqrt_beta * std
        return float(ucb.min().item()), X_evaluated

    def _min_lcb_sample(
        self,
        strategy,
        X_evaluated: torch.Tensor,
        sqrt_beta: float,
    ) -> float:
        """Compute min LCB via random sampling + evaluated points.

        Following the reference implementation of Makarova et al. (2022):
        generates random candidates in the transformed input space, merges
        with observed points, and evaluates LCB at all points in batches.

        Args:
            strategy: A fitted BotorchStrategy.
            X_evaluated: Transformed evaluated input tensor.
            sqrt_beta: Square root of beta for LCB computation.

        Returns:
            The minimum LCB value found.
        """
        bounds = strategy.domain.inputs.get_bounds(
            specs=strategy.input_preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)
        n_transformed_dims = len(lower)

        # Random samples in transformed space
        X_random = lower + (upper - lower) * torch.rand(
            self.n_samples_lcb, n_transformed_dims, **tkwargs
        )
        # Merge with evaluated points
        X_lcb_candidates = torch.cat([X_random, X_evaluated], dim=0)

        # Compute LCB in batches to avoid memory blowup
        min_lcb = float("inf")
        with torch.no_grad():
            for start in range(0, X_lcb_candidates.shape[0], self.batch_size):
                X_batch = X_lcb_candidates[start : start + self.batch_size]
                posterior_batch = strategy.model.posterior(X_batch)
                mean_batch = posterior_batch.mean.squeeze(-1)
                std_batch = posterior_batch.variance.squeeze(-1).sqrt()
                lcb_batch = mean_batch - sqrt_beta * std_batch
                batch_min = float(lcb_batch.min().item())
                if batch_min < min_lcb:
                    min_lcb = batch_min

        return min_lcb

    def _min_lcb_optimize(
        self,
        strategy,
        experiments: pd.DataFrame,
        sqrt_beta: float,
    ) -> float:
        """Compute min LCB via BoTorch acquisition function optimization.

        Uses the strategy's own acquisition optimizer to minimize LCB over the
        domain. This handles continuous, categorical, discrete, and mixed
        variable types identically to the main BO loop. More principled than
        random sampling but slower.

        Args:
            strategy: A fitted BotorchStrategy.
            experiments: Current experiments DataFrame.
            sqrt_beta: Square root of beta for LCB computation.

        Returns:
            The minimum LCB value found.
        """
        neg_lcb = _NegLowerConfidenceBound(
            model=strategy.model, sqrt_beta=float(sqrt_beta)
        )
        candidates = strategy.acqf_optimizer.optimize(
            candidate_count=1,
            acqfs=[neg_lcb],
            domain=strategy.domain,
            experiments=experiments,
        )

        # Evaluate LCB at the optimized candidate
        transformed_candidate = strategy.domain.inputs.transform(
            candidates,
            strategy.input_preprocessing_specs,
        )
        X_best = torch.from_numpy(transformed_candidate.values).to(**tkwargs)
        with torch.no_grad():
            posterior_best = strategy.model.posterior(X_best)
            mean_best = posterior_best.mean.squeeze(-1)
            std_best = posterior_best.variance.squeeze(-1).sqrt()

        return float((mean_best - sqrt_beta * std_best).item())
