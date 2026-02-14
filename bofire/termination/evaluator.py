"""Termination evaluators that compute metrics for termination conditions.

These evaluators work with BotorchStrategy instances to compute the various
metrics used by termination conditions (e.g., UCB-LCB regret bound).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

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

    This provides an upper bound on the simple regret of the current best
    recommendation. When this bound becomes small (relative to noise), further
    optimization is unlikely to yield significant improvement.

    If the strategy uses qUCB as its acquisition function, the same beta value
    is used for UCB/LCB computation. Otherwise, uses the default beta (0.2).

    Reference:
        Makarova et al. (2022): "Automatic Termination for Hyperparameter
        Optimization" (AutoML 2022)
    """

    fallback_noise_variance: float = 1e-4
    fallback_beta: float = 0.2

    def _get_beta(self, strategy) -> float:
        """Get beta for UCB/LCB computation.

        If the strategy uses qUCB, returns its beta value.
        Otherwise, returns the default beta (0.2).

        Args:
            strategy: The BotorchStrategy.

        Returns:
            The beta value for UCB/LCB bounds.
        """
        # Try to get beta from qUCB acquisition function
        try:
            acqf = strategy.acquisition_function
            if hasattr(acqf, "beta"):
                return acqf.beta
        except AttributeError:
            pass

        return self.fallback_beta

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Compute the UCB-LCB regret bound.

        Finds min_x LCB(x) over the domain using the strategy's own acquisition
        optimizer (which handles continuous, categorical, discrete, and mixed
        variable types identically to the main BO loop). Computes min UCB over
        evaluated points directly.

        Args:
            strategy: A fitted BotorchStrategy.
            experiments: Current experiments.
            iteration: Current iteration.

        Returns:
            Dictionary with 'regret_bound', 'min_ucb_evaluated', 'min_lcb_domain',
            and 'estimated_noise_variance'.
        """
        if not strategy.is_fitted or strategy.model is None:
            return {}

        if len(experiments) < 2:
            return {}

        # Compute beta (from qUCB if used, otherwise default 0.2)
        beta = self._get_beta(strategy)
        sqrt_beta = np.sqrt(beta)

        # --- min UCB over evaluated points ---
        evaluated_inputs = experiments[strategy.domain.inputs.get_keys()]
        transformed_evaluated = strategy.domain.inputs.transform(
            evaluated_inputs,
            strategy.input_preprocessing_specs,
        )
        X_evaluated = torch.from_numpy(transformed_evaluated.values).to(**tkwargs)

        with torch.no_grad():
            posterior_evaluated = strategy.model.posterior(X_evaluated)
            mean_evaluated = posterior_evaluated.mean.squeeze(-1)
            std_evaluated = posterior_evaluated.variance.squeeze(-1).sqrt()

        ucb_evaluated = mean_evaluated + sqrt_beta * std_evaluated
        min_ucb_evaluated = float(ucb_evaluated.min().item())

        # --- min LCB over domain using the strategy's acquisition optimizer ---
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
        min_lcb_domain = float((mean_best - sqrt_beta * std_best).item())

        # Regret bound = min_UCB(evaluated) - min_LCB(domain)
        regret_bound = max(0.0, min_ucb_evaluated - min_lcb_domain)

        # Estimate noise variance from GP likelihood
        try:
            estimated_noise_var = strategy.model.likelihood.noise.item()
        except Exception:
            estimated_noise_var = self.fallback_noise_variance

        return {
            "regret_bound": regret_bound,
            "min_ucb_evaluated": min_ucb_evaluated,
            "min_lcb_domain": min_lcb_domain,
            "estimated_noise_variance": estimated_noise_var,
            "beta": beta,
        }
