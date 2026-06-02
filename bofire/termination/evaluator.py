"""Termination evaluators that compute metrics for termination conditions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.models.model import Model

from bofire.utils.torch_tools import tkwargs


class TerminationEvaluator(ABC):
    """Base class for termination evaluators.

    Computes metrics from a BO strategy that termination conditions use to
    decide whether to stop.

    Args:
        delta: Confidence parameter for the GP-UCB beta formula (Srinivas et
            al., 2010). Controls how optimistic the confidence bounds are.
            Smaller values give larger beta (wider bounds). Default ``0.1``.
        beta_scale: Scaling factor applied to the GP-UCB beta.  Makarova et
            al. use ``0.2``; Ishibashi et al. use ``1.0``.
        beta_log_multiplier: Multiplier inside the log term of the beta
            formula.  Default ``2.0``.
        beta_log_denominator: Denominator inside the log term of the beta
            formula.  Default ``6.0``.
        beta_min: Floor on beta to avoid degenerate near-zero bounds.
            Default ``0.01``.
        beta_t_offset: If set, the beta formula uses
            ``t = num_observed - beta_t_offset`` instead of the total
            observation count.  Useful when papers index beta by BO iteration
            rather than total evaluations (e.g. excluding the random init).
    """

    def __init__(
        self,
        delta: float = 0.1,
        beta_scale: float = 0.2,
        beta_log_multiplier: float = 2.0,
        beta_log_denominator: float = 6.0,
        beta_min: float = 0.01,
        beta_t_offset: Optional[int] = None,
    ):
        # GP-UCB beta parameters (Srinivas et al., 2010).
        self.delta = delta
        self.beta_scale = beta_scale
        self.beta_log_multiplier = beta_log_multiplier
        self.beta_log_denominator = beta_log_denominator
        self.beta_min = beta_min
        # If set, use ``t = num_observed - beta_t_offset`` in the beta formula
        # (for papers that index beta by BO iteration rather than observation count).
        self.beta_t_offset = beta_t_offset

    def _compute_beta(self, dimensionality: int, num_observed: int) -> float:
        """Compute beta for UCB/LCB via the GP-UCB formula.

        ``beta_t = beta_scale * 2 * log(d * t^2 * pi^2 / (6 * delta))``.
        Makarova et al. use ``beta_scale=0.2``; Ishibashi et al. use ``1.0``.
        """
        t = num_observed
        if self.beta_t_offset is not None:
            t = max(1, num_observed - self.beta_t_offset)
        beta = (
            self.beta_scale
            * self.beta_log_multiplier
            * np.log(
                dimensionality
                * t**2
                * np.pi**2
                / (self.beta_log_denominator * self.delta)
            )
        )
        return max(beta, self.beta_min)

    @staticmethod
    def get_output_scale(model: Model) -> float:
        """Return the output std from a ``Standardize`` outcome transform, or 1.0.

        BoTorch models with ``Standardize`` return posteriors in original scale,
        while the reference stopping criteria (Ishibashi / Makarova) assume
        standardized quantities. Callers can divide by this value to convert
        back when needed.
        """
        if hasattr(model, "outcome_transform"):
            try:
                return float(model.outcome_transform.stdvs.item())
            except Exception:
                pass
        return 1.0

    @staticmethod
    def _min_ucb_at_points(
        model: Model,
        X: torch.Tensor,
        sqrt_beta: float,
    ) -> float:
        """Return min UCB = mu + sqrt(beta)*sigma over ``X``."""
        ucb = UpperConfidenceBound(model=model, beta=sqrt_beta**2)
        with torch.no_grad():
            return float(ucb(X.unsqueeze(-2)).min().item())

    @staticmethod
    def _min_lcb_by_sampling(
        model: Model,
        X_evaluated: torch.Tensor,
        sqrt_beta: float,
        bounds_lower: torch.Tensor,
        bounds_upper: torch.Tensor,
        n_samples: int = 2000,
        batch_size: int = 512,
    ) -> float:
        """Return min LCB = mu - sqrt(beta)*sigma via random sampling."""
        n_dims = bounds_lower.shape[0]
        X_random = bounds_lower + (bounds_upper - bounds_lower) * torch.rand(
            n_samples, n_dims, **tkwargs
        )
        X_candidates = torch.cat([X_random, X_evaluated], dim=0)

        min_lcb = float("inf")
        with torch.no_grad():
            for start in range(0, X_candidates.shape[0], batch_size):
                X_batch = X_candidates[start : start + batch_size]
                post = model.posterior(X_batch)
                m = post.mean.squeeze(-1)
                s = post.variance.squeeze(-1).sqrt()
                lcb = m - sqrt_beta * s
                batch_min = float(lcb.min().item())
                if batch_min < min_lcb:
                    min_lcb = batch_min
        return min_lcb

    def _min_lcb_optimize(
        self,
        strategy,
        experiments: pd.DataFrame,
        sqrt_beta: float,
    ) -> float:
        """Minimise LCB over the domain via the strategy's acqf optimizer."""
        neg_lcb = UpperConfidenceBound(
            model=strategy.model, beta=float(sqrt_beta) ** 2, maximize=False
        )
        candidates = strategy.acqf_optimizer.optimize(
            candidate_count=1,
            acqfs=[neg_lcb],
            domain=strategy.domain,
            experiments=experiments,
        )

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

    def _ucb_lcb_regret_bound(
        self,
        model: Model,
        X_evaluated: torch.Tensor,
        sqrt_beta: float,
        bounds_lower: torch.Tensor,
        bounds_upper: torch.Tensor,
        n_samples: int = 2000,
        batch_size: int = 512,
        strategy=None,
        experiments: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Upper bound on simple regret via the UCB-LCB gap.

        Returns ``(regret_bound, min_ucb, min_lcb)`` with
        ``regret_bound = max(0, min_UCB(evaluated) - min_LCB(domain))``.
        """
        min_ucb = TerminationEvaluator._min_ucb_at_points(
            model,
            X_evaluated,
            sqrt_beta,
        )

        if (
            self.lcb_method == "optimize"
            and strategy is not None
            and experiments is not None
        ):
            min_lcb = self._min_lcb_optimize(strategy, experiments, sqrt_beta)
        else:
            min_lcb = TerminationEvaluator._min_lcb_by_sampling(
                model,
                X_evaluated,
                sqrt_beta,
                bounds_lower,
                bounds_upper,
                n_samples,
                batch_size,
            )
        return max(0.0, min_ucb - min_lcb), min_ucb, min_lcb

    @abstractmethod
    def evaluate(
        self,
        strategy,  # BotorchStrategy
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Return a dict of termination metrics for the current iteration."""
        pass
