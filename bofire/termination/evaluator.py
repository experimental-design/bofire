"""Termination evaluators that compute metrics for termination conditions."""

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    _log_ei_helper,
)
from botorch.acquisition.analytic import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm

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
        ucb = UpperConfidenceBound(model=model, beta=sqrt_beta ** 2)
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
            model, X_evaluated, sqrt_beta,
        )

        if self.lcb_method == "optimize" and strategy is not None and experiments is not None:
            min_lcb = self._min_lcb_optimize(strategy, experiments, sqrt_beta)
        else:
            min_lcb = TerminationEvaluator._min_lcb_by_sampling(
                model, X_evaluated, sqrt_beta, bounds_lower, bounds_upper,
                n_samples, batch_size,
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


# TODO: replace with `from botorch.acquisition.analytic import LogExpectedImprovementPerCost`
#       once the BoTorch PR (wgst/botorch feature/log-expected-improvement-with-cost) is merged.
class LogExpectedImprovementPerCost(AnalyticAcquisitionFunction):
    r"""Single-outcome Log Expected Improvement per unit cost (analytic).

    Computes the log expected improvement adjusted for the cost of evaluating
    the candidate point:

        LogEIC(x) = LogEI(x; best_f) - alpha * log(c(x))

    where ``LogEI`` is the log expected improvement (Ament et al., 2023) and
    ``c(x)`` is the evaluation cost at ``x``.

    This is the acquisition function underlying the cost-aware stopping rule
    of Xie et al. (2025): stop when ``max_x LogEIC(x) + log(lambda)`` is
    non-positive, i.e. no candidate's expected improvement exceeds its cost
    scaled by the exchange rate ``lambda``.

    Args:
        model: A fitted single-output GP model.
        best_f: Best observed function value (EI baseline). Scalar or
            ``(batch_shape,)`` tensor.
        cost_callable: ``c(X: Tensor[..., d]) -> Tensor[...]`` — evaluation
            cost at each candidate point. Must return strictly positive values.
        alpha: Cost exponent in ``c(x)^alpha``. ``1.0`` (default) matches the
            primary formulation of Xie et al. (2025).
        posterior_transform: Optional posterior transform.
        maximize: If ``True``, treat the problem as maximisation.

    References:
        Ament et al. (2023): "Unexpected Improvements to Expected Improvement
            for Bayesian Optimization" (NeurIPS 2023).
        Xie et al. (2025): "Cost-Aware Stopping for Bayesian Optimization"
            (arXiv:2507.12453).
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, torch.Tensor],
        cost_callable: Callable[[torch.Tensor], torch.Tensor],
        alpha: float = 1.0,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = False,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.cost_callable = cost_callable
        self.alpha = alpha

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate LogEIC at ``X``.

        Args:
            X: ``(b1 x ... bk) x 1 x d``-dim batched tensor of candidate points.

        Returns:
            ``(b1 x ... bk)``-dim tensor of LogEIC values.
        """
        posterior = self.model.posterior(
            X.squeeze(-2), posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-1)
        sigma = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt()
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        log_ei = _log_ei_helper(u) + sigma.log()

        costs = self.cost_callable(X.squeeze(-2))
        log_cost = costs.clamp(min=1e-12).log()
        return log_ei - self.alpha * log_cost



class UCBLCBRegretEvaluator(TerminationEvaluator):
    """Evaluator for the UCB-LCB regret bound (Makarova et al., 2022).

    The bound is ``min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)``
    using the GP-UCB formulation (Srinivas et al., 2010).  Stopping is
    triggered when this gap is small — meaning the best candidate already
    evaluated is nearly as good as the best anywhere in the domain.

    Single-output minimisation only.

    Args:
        delta: Confidence parameter for the GP-UCB beta formula. Default ``0.1``.
        beta_scale: Scaling factor for beta. Makarova et al. use ``0.2``.
        beta_log_multiplier: Multiplier inside the log term of beta. Default ``2.0``.
        beta_log_denominator: Denominator inside the log term of beta. Default ``6.0``.
        beta_min: Floor on beta. Default ``0.01``.
        beta_t_offset: If set, indexes beta by BO iteration rather than total
            observation count (``t = num_observed - beta_t_offset``).
        fallback_noise_variance: Noise variance used when it cannot be read
            from the GP likelihood (e.g. deterministic surrogates). Default ``1e-4``.
        n_samples_lcb: Number of random domain points used when
            ``lcb_method="sample"`` to approximate the minimum LCB over the
            domain. Default ``2000``.
        batch_size: Batch size for GP posterior evaluation during sampling.
            Default ``512``.
        lcb_method: How the domain-wide minimum LCB is found.
            ``"sample"`` (default) draws ``n_samples_lcb`` random points —
            matches the reference implementation.  ``"optimize"`` uses
            BoFire's acquisition function optimizer for higher accuracy.

    References:
        Makarova et al. (2022): "Automatic Termination for Hyperparameter
            Optimization" (AutoML 2022).
        Srinivas et al. (2010): "Gaussian Process Optimization in the Bandit
            Setting: No Regret and Experimental Design" (ICML 2010).
    """

    def __init__(
        self,
        delta: float = 0.1,
        beta_scale: float = 0.2,
        beta_log_multiplier: float = 2.0,
        beta_log_denominator: float = 6.0,
        beta_min: float = 0.01,
        beta_t_offset: Optional[int] = None,
        fallback_noise_variance: float = 1e-4,
        n_samples_lcb: int = 2000,
        batch_size: int = 512,
        lcb_method: Literal["sample", "optimize"] = "sample",
    ):
        super().__init__(
            delta=delta,
            beta_scale=beta_scale,
            beta_log_multiplier=beta_log_multiplier,
            beta_log_denominator=beta_log_denominator,
            beta_min=beta_min,
            beta_t_offset=beta_t_offset,
        )
        self.fallback_noise_variance = fallback_noise_variance
        self.n_samples_lcb = n_samples_lcb
        self.batch_size = batch_size
        self.lcb_method = lcb_method

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Return regret-bound metrics, or an empty dict when not applicable."""
        if not strategy.is_fitted or strategy.model is None:
            return {}

        if len(experiments) < 2:
            return {}

        if strategy.model.num_outputs != 1:
            return {}

        input_keys = strategy.domain.inputs.get_keys()
        dimensionality = len(input_keys)
        num_observed = len(experiments)
        beta = self._compute_beta(dimensionality, num_observed)
        sqrt_beta = np.sqrt(beta)

        evaluated_inputs = experiments[input_keys]
        transformed_evaluated = strategy.domain.inputs.transform(
            evaluated_inputs,
            strategy.input_preprocessing_specs,
        )
        X_evaluated = torch.from_numpy(transformed_evaluated.values).to(**tkwargs)

        bounds = strategy.domain.inputs.get_bounds(
            specs=strategy.input_preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        regret_bound, min_ucb_evaluated, min_lcb_domain = self._ucb_lcb_regret_bound(
            strategy.model, X_evaluated, sqrt_beta, lower, upper,
            self.n_samples_lcb, self.batch_size, strategy=strategy,
            experiments=experiments,
        )

        # Noise variance from the GP likelihood (used by conditions to
        # compute the termination threshold).
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


class ExpMinRegretGapEvaluator(TerminationEvaluator):
    """Evaluator for the expected minimum regret gap (Ishibashi et al., 2023).

    Computes a stopping value that upper-bounds the change in expected minimum
    simple regret between consecutive BO iterations:

        value_t = delta_f + ei_diff + kappa * sqrt(KL / 2)

    where ``delta_f`` is the change in the GP mean at the incumbent,
    ``ei_diff`` is the expected improvement from switching incumbents,
    ``kappa`` is the UCB-LCB regret bound from the previous model, and
    ``KL`` is the KL divergence of the old GP prior vs. the updated
    posterior at the newly observed point.

    Stateful: stores the previous GP model and incumbent index between calls.
    The first call always returns an empty dict. Single-output minimisation only.

    Args:
        delta: Confidence parameter for the GP-UCB beta formula and the
            adaptive threshold. Default ``0.1``.
        beta_scale: Scaling factor for beta. Ishibashi et al. use ``1.0``
            (no extra scaling).
        beta_log_multiplier: Multiplier inside the log term of beta. Default ``2.0``.
        beta_log_denominator: Denominator inside the log term of beta. Default ``6.0``.
        beta_min: Floor on beta. Default ``0.01``.
        beta_t_offset: If set, indexes beta by BO iteration rather than total
            observation count (``t = num_observed - beta_t_offset``).
        n_samples_lcb: Random domain points for the min-LCB estimate used in
            ``kappa``. Default ``1000``.
        batch_size: Batch size for GP posterior evaluation during sampling.
            Default ``512``.
        threshold_mode: How the stopping threshold is computed.
            ``"adaptive"`` uses the theoretically motivated threshold from the
            GP noise and posterior variances (eq. 16 in the paper).
            ``"median"`` uses ``rate * median`` of the first ``start_timing``
            stopping values. ``"adaptive_median"`` (default) combines both —
            stops when either threshold is exceeded.
        start_timing: Number of stopping values to collect before the median
            threshold can be computed. Default ``10``.
        rate: Fraction of the median used as the median threshold.
            Default ``0.1``.
        noise_var_override: If set, overrides the GP's learned noise variance.
            Set to ``1e-6`` for noise-free synthetic benchmarks to match the
            reference implementation.
        lcb_method: How the domain-wide minimum LCB (``kappa``) is found.
            ``"sample"`` (default) or ``"optimize"``.

    Reference:
        Ishibashi et al. (2023): "A stopping criterion for Bayesian optimization
            by the gap of expected minimum simple regrets" (AISTATS 2023).
    """

    def __init__(
        self,
        delta: float = 0.1,
        beta_scale: float = 1.0,  # Ishibashi et al. use no extra scaling.
        beta_log_multiplier: float = 2.0,
        beta_log_denominator: float = 6.0,
        beta_min: float = 0.01,
        beta_t_offset: Optional[int] = None,
        n_samples_lcb: int = 1000,
        batch_size: int = 512,
        threshold_mode: Literal[
            "adaptive", "median", "adaptive_median"
        ] = "adaptive_median",
        start_timing: int = 10,
        rate: float = 0.1,
        noise_var_override: Optional[float] = None,
        lcb_method: Literal["sample", "optimize"] = "sample",
    ):
        super().__init__(
            delta=delta,
            beta_scale=beta_scale,
            beta_log_multiplier=beta_log_multiplier,
            beta_log_denominator=beta_log_denominator,
            beta_min=beta_min,
            beta_t_offset=beta_t_offset,
        )
        self.n_samples_lcb = n_samples_lcb
        self.batch_size = batch_size
        self.threshold_mode = threshold_mode
        self.start_timing = start_timing
        self.rate = rate
        self.lcb_method = lcb_method
        # If set, overrides the GP's learned noise variance (the reference
        # implementation uses ~1e-6 for noise-free synthetic functions).
        self.noise_var_override = noise_var_override

        # Internal state persisted across evaluate() calls.
        self._prev_model: Optional[Model] = None
        self._prev_incumbent_idx: Optional[int] = None
        self._prev_n_experiments: int = 0
        self._prev_input_preprocessing_specs: Optional[Dict] = None
        self._seq_values: List[float] = []

    @staticmethod
    def _get_noise_variance(model) -> float:
        """Return GP noise variance in the original (un-standardized) output scale.

        BoTorch models with a ``Standardize`` outcome transform learn noise in
        standardized space, but posteriors are returned in the original scale,
        so the noise must be un-standardized to match:
        ``noise_var_original = noise_var_standardized * y_std**2``.
        """
        try:
            noise_var = model.likelihood.noise.item()
        except Exception:
            return 1e-4

        if hasattr(model, "outcome_transform"):
            try:
                y_std = model.outcome_transform.stdvs.item()
                noise_var = noise_var * y_std**2
            except Exception:
                pass

        return noise_var

    def _calc_kl_qp_fast(
        self,
        old_mean: float,
        old_var: float,
        new_output: float,
        noise_var: float,
    ) -> float:
        """Closed-form KL(q || p) for a single Gaussian observation."""
        precision = 1.0 / noise_var
        k = old_var
        m = old_mean
        trace_term = k / (k + 1.0 / precision)
        logdet_term = np.log(1.0 + precision * k)
        se_term = k / ((k + 1.0 / precision) ** 2) * (new_output - m) ** 2
        kl = 0.5 * (-trace_term + logdet_term + se_term)
        return max(0.0, float(kl))

    def _compute_ei_diff(
        self,
        model: Model,
        X_new_incumbent: torch.Tensor,
        X_old_incumbent: torch.Tensor,
    ) -> float:
        """Expected improvement from switching incumbents.

        If the incumbent changed, computes E[max(f(x*_new) - f(x*_old), 0)]
        under the new GP's joint posterior at the two incumbent locations.
        """
        X_pair = torch.cat([X_new_incumbent, X_old_incumbent], dim=0)
        with torch.no_grad():
            posterior = model.posterior(X_pair)
            mu = posterior.mean.squeeze(-1)
            cov = posterior.distribution.covariance_matrix

        g = float((mu[0] - mu[1]).item())
        var_diff = float(
            (cov[0, 0] - 2 * cov[0, 1] + cov[1, 1]).item()
        )

        if var_diff <= 0:
            # No uncertainty about the difference between the two incumbents.
            beta_val = 0.0
            pdf_val = np.sqrt(1.0 / (2.0 * np.pi))
            cdf_val = 1.0
        else:
            beta_val = np.sqrt(var_diff)
            u = g / beta_val
            pdf_val = float(norm.pdf(u))
            cdf_val = float(norm.cdf(u))

        return float(beta_val * pdf_val + g * cdf_val)

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Return regret-gap metrics, or an empty dict on the first call / when not applicable."""
        if not strategy.is_fitted or strategy.model is None:
            return {}
        if len(experiments) < 2:
            return {}
        if strategy.model.num_outputs != 1:
            return {}

        input_keys = strategy.domain.inputs.get_keys()
        output_key = strategy.domain.outputs.get_keys()[0]
        dimensionality = len(input_keys)
        n_experiments = len(experiments)

        incumbent_idx = int(experiments[output_key].idxmin())

        # First call: save state and return empty.
        if self._prev_model is None or n_experiments <= self._prev_n_experiments:
            self._save_state(strategy, experiments, incumbent_idx)
            return {}

        # Assume one new point added since the last call; take the last.
        new_point_idx = n_experiments - 1
        y_new = float(experiments[output_key].iloc[new_point_idx])

        preprocessing_specs = strategy.input_preprocessing_specs
        all_inputs = experiments[input_keys]
        transformed = strategy.domain.inputs.transform(
            all_inputs, preprocessing_specs
        )
        X_all = torch.from_numpy(transformed.values).to(**tkwargs)

        X_prev = X_all[: self._prev_n_experiments]
        x_new = X_all[[new_point_idx]]
        x_incumbent_new = X_all[[incumbent_idx]]
        x_incumbent_old = X_all[[self._prev_incumbent_idx]]

        # Use the old data count for beta, matching the reference implementation.
        sqrt_beta = np.sqrt(
            self._compute_beta(dimensionality, self._prev_n_experiments)
        )

        with torch.no_grad():
            old_posterior = self._prev_model.posterior(x_new)
            old_mean = float(old_posterior.mean.squeeze().item())
            old_var = float(old_posterior.variance.squeeze().item())

        noise_var = (
            self.noise_var_override
            if self.noise_var_override is not None
            else self._get_noise_variance(strategy.model)
        )

        # BoTorch with Standardize: noise_var is in standardized space but
        # posterior returns original-scale predictions. Un-standardize noise
        # so all quantities are in the same scale (matching reference which
        # uses normalize_Y=False where everything is natively in raw scale).
        y_std = self.get_output_scale(strategy.model)
        noise_var_original = noise_var * y_std**2

        kl = self._calc_kl_qp_fast(old_mean, old_var, y_new, noise_var_original)

        # delta_f: change in predicted incumbent value.
        with torch.no_grad():
            old_mu_at_old_inc = float(
                self._prev_model.posterior(x_incumbent_old).mean.squeeze().item()
            )
            new_mu_at_new_inc = float(
                strategy.model.posterior(x_incumbent_new).mean.squeeze().item()
            )
        delta_f = abs(old_mu_at_old_inc - new_mu_at_new_inc)

        # kappa: UCB-LCB regret bound using the old model.
        bounds = strategy.domain.inputs.get_bounds(
            specs=preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        kappa, _, _ = self._ucb_lcb_regret_bound(
            self._prev_model, X_prev, sqrt_beta, lower, upper,
            self.n_samples_lcb, self.batch_size, strategy=strategy,
            experiments=experiments,
        )

        # ei_diff: expected improvement from switching incumbents.
        if incumbent_idx == self._prev_incumbent_idx:
            ei_diff = 0.0
        else:
            ei_diff = self._compute_ei_diff(
                strategy.model, x_incumbent_new, x_incumbent_old
            )

        stopping_value = delta_f + ei_diff + kappa * np.sqrt(0.5 * kl)

        threshold_adaptive = None
        threshold_median = None

        if self.threshold_mode in ("adaptive", "adaptive_median"):
            # Adaptive threshold in raw scale (reference uses normalize_Y=False).
            threshold_adaptive = self._compute_threshold_adaptive(
                self._prev_model, x_incumbent_new,
                old_var,
                noise_var_original, kappa,
            )

        self._seq_values.append(stopping_value)
        if self.threshold_mode in ("median", "adaptive_median"):
            threshold_median = self._compute_threshold_median(
                self._seq_values, self.start_timing, self.rate,
            )

        self._save_state(strategy, experiments, incumbent_idx)

        return {
            "stopping_value": stopping_value,
            "delta_f": delta_f,
            "ei_diff": ei_diff,
            "kappa": kappa,
            "kl_divergence": kl,
            "threshold_adaptive": threshold_adaptive,
            "threshold_median": threshold_median,
            "noise_variance": noise_var_original,
            "seq_values": list(self._seq_values),
        }

    def _save_state(
        self,
        strategy,
        experiments: pd.DataFrame,
        incumbent_idx: int,
    ) -> None:
        """Save current state for comparison at the next evaluate() call."""
        self._prev_model = strategy.model
        self._prev_n_experiments = len(experiments)
        self._prev_incumbent_idx = incumbent_idx
        self._prev_input_preprocessing_specs = strategy.input_preprocessing_specs

    def _compute_threshold_adaptive(
        self,
        old_model: Model,
        x_incumbent_new: torch.Tensor,
        var_old_new: float,
        noise_var: float,
        kappa: float,
    ) -> Optional[float]:
        """Adaptive threshold from Ishibashi et al. (2023), eq. (16).

        All inputs are in raw Y scale (the reference implementation uses
        ``normalize_Y=False``).

        Args:
            old_model: The previous GP model.
            x_incumbent_new: Transformed input of the new incumbent ``(1, d)``.
            var_old_new: Old GP's predictive variance at the new data point.
            noise_var: GP noise variance.
            kappa: UCB-LCB regret bound from the old GP.

        Returns:
            The adaptive threshold, or ``None`` when the denominator is
            non-positive.
        """
        c = np.sqrt(-2.0 * np.log(self.delta))
        precision = 1.0 / noise_var
        with torch.no_grad():
            var_old_incumbent = float(
                old_model.posterior(x_incumbent_new).variance.squeeze().item()
            )

        denom = np.sqrt(precision) * (var_old_new + noise_var)
        if denom <= 0:
            return None
        eps1 = np.sqrt(var_old_incumbent) * np.sqrt(var_old_new) * c / denom
        eps2 = (kappa / 2.0) * np.sqrt(var_old_new) * c / denom
        return float(eps1 + eps2)

    @staticmethod
    def _compute_threshold_median(
        seq_values: List[float],
        start_timing: int,
        rate: float,
    ) -> Optional[float]:
        """``rate * median(seq_values[:start_timing])``, or ``None`` if not enough values."""
        if len(seq_values) <= start_timing:
            return None
        return float(rate * np.median(seq_values[:start_timing]))

    def dumps(self) -> str:
        """Dumps the internal state to a base64 encoded pickle string.

        Analogous to ``BotorchSurrogate.dumps``, so the state can be embedded
        in JSON.
        """
        state = {
            "prev_model": self._prev_model,
            "prev_incumbent_idx": self._prev_incumbent_idx,
            "prev_n_experiments": self._prev_n_experiments,
            "prev_input_preprocessing_specs": self._prev_input_preprocessing_specs,
            "seq_values": self._seq_values,
        }
        buffer = io.BytesIO()
        torch.save(state, buffer)
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str) -> None:
        """Loads the internal state from a base64 encoded pickle string produced by ``dumps``.

        Args:
            data: Base64-encoded string previously returned by :meth:`dumps`.
        """
        buffer = io.BytesIO(base64.b64decode(data.encode()))
        state = torch.load(buffer, weights_only=False)
        self._prev_model = state["prev_model"]
        self._prev_incumbent_idx = state["prev_incumbent_idx"]
        self._prev_n_experiments = state["prev_n_experiments"]
        self._prev_input_preprocessing_specs = state[
            "prev_input_preprocessing_specs"
        ]
        self._seq_values = state["seq_values"]


class LogEIPCEvaluator(TerminationEvaluator):
    """Cost-aware stopping criterion (Xie et al., 2025).

    Stops when the maximum log expected-improvement-per-cost over the domain
    falls to zero or below — meaning no candidate point's expected improvement
    is worth its evaluation cost:

        stop when  max_x [ LogEI(x) - alpha * log(c(x)) - log(lambda_cost) ] <= 0

    Equivalently: stop when ``max_x EI(x) <= lambda_cost * c(x)^alpha``.

    Ideal for chemical experiments where reagent, time, or equipment costs
    matter. Single-output minimisation only.

    Args:
        lambda_cost: Exchange rate between improvement and cost.  Stopping
            is triggered when the best expected improvement no longer exceeds
            ``lambda_cost * c(x)``.  Higher values favour earlier stopping.
            Think of it as "the minimum expected improvement per unit cost
            that justifies one more experiment".  With ``cost_value=1.0`` this
            is directly a threshold on EI in objective units (e.g. 0.05 means
            "stop when EI < 5% of objective range").
        cost_callable: Optional function ``c(X: Tensor[n, d]) -> Tensor[n]``
            that returns a per-point cost for each candidate in transformed
            input space.  Enables input-dependent costs (e.g. reaction time
            proportional to temperature: ``lambda X: 1.0 + 3.0 * X[:, 0]``).
            Takes priority over ``cost_column`` and ``cost_value``.
            Not serialisable — use the evaluator directly rather than via
            ``LogEIPCCondition`` when input-dependent costs are needed.
        cost_column: Name of the column in the experiments DataFrame that
            records the actual cost of each completed experiment.  The running
            mean is used as the cost estimate for future candidates.  Takes
            priority over ``cost_value`` when the column is present.
        cost_value: Fixed scalar cost per experiment used when neither
            ``cost_callable`` nor ``cost_column`` is provided.  Default ``1.0``
            (uniform cost — criterion degenerates to a pure LogEI threshold).
        alpha: Exponent applied to the cost: ``c(x)^alpha``.  ``1.0`` (default)
            matches the paper.  Values < 1 reduce the influence of cost.
        n_samples: Number of random domain points used when
            ``search_method="sample"`` to approximate the max LogEIPC.
            Increase for higher dimensions where default coverage may be sparse.
        batch_size: Batch size for evaluating the GP posterior during sampling,
            used to control memory usage.
        search_method: How the maximum LogEIPC over the domain is found.
            ``"sample"`` (default) draws ``n_samples`` random points — fast,
            robust, matches the reference implementation.  ``"optimize"`` uses
            BoFire's acquisition function optimizer via ``_LogEIPCWithCost`` —
            more accurate in high dimensions but slower and can misfire if the
            optimizer converges to a local maximum.
        cost_model: How cost is estimated when ``cost_column`` is provided.
            ``"mean"`` (default) uses the scalar mean of past costs — simple
            and backward-compatible.  ``"gp"`` fits a ``SingleTaskGP`` to the
            observed ``(x, log_cost)`` pairs and uses its posterior mean as a
            per-point cost callable, capturing spatial variation in cost.
            Matches the paper's "unknown cost" approach.  Ignored when
            ``cost_callable`` is set directly.

    References:
        Ament et al. (2023): "Unexpected Improvements to Expected Improvement
            for Bayesian Optimization" (NeurIPS 2023) — LogEIPC acquisition
            function (log EI per cost).
        Xie et al. (2025): "Cost-Aware Stopping for Bayesian Optimization"
            (arXiv:2507.12453) — LogEIPC stopping rule (max LogEIPC ≤ 0)
            grounded in Pandora's Box theory.
    """

    def __init__(
        self,
        lambda_cost: float = 1.0,
        cost_column: Optional[str] = None,
        cost_value: float = 1.0,
        alpha: float = 1.0,
        n_samples: int = 2000,
        batch_size: int = 512,
        cost_callable: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        search_method: Literal["sample", "optimize"] = "sample",
        cost_model: Literal["mean", "gp"] = "mean",
    ):
        # TerminationEvaluator base class holds beta parameters for UCB/LCB;
        # LogEIPC does not use them but we call super().__init__() for
        # get_output_scale and other shared utilities.
        super().__init__()
        self.lambda_cost = lambda_cost
        self.cost_column = cost_column
        self.cost_value = cost_value
        self.alpha = alpha
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.cost_callable = cost_callable
        self.search_method = search_method
        self.cost_model = cost_model

    def _fit_cost_gp(
        self,
        strategy,
        experiments: pd.DataFrame,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Fit a GP to observed (X, log_cost) pairs and return a cost callable.

        Matches the "unknown cost" approach from Xie et al., 2025: fits a
        ``SingleTaskGP`` on log-costs and returns a callable that predicts
        ``exp(GP_posterior_mean(x))`` at any candidate point.

        Returns ``None`` when there are too few observations or all costs are
        non-positive (log undefined).
        """
        if self.cost_column is None:
            return None

        input_keys = strategy.domain.inputs.get_keys()
        valid = experiments[[*input_keys, self.cost_column]].dropna()
        costs = valid[self.cost_column].values.astype(float)

        if len(valid) < 2 or (costs <= 0).any():
            return None

        transformed = strategy.domain.inputs.transform(
            valid[input_keys], strategy.input_preprocessing_specs
        )
        X = torch.tensor(transformed.values, **tkwargs)
        log_c = torch.tensor(np.log(costs), **tkwargs).unsqueeze(-1)

        cost_gp = SingleTaskGP(X, log_c)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(cost_gp.likelihood, cost_gp))
        cost_gp.eval()

        def cost_callable(X_cands: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return torch.exp(cost_gp.posterior(X_cands).mean.squeeze(-1))

        return cost_callable

    def _get_cost_estimate(self, experiments: pd.DataFrame) -> float:
        """Return the cost estimate for future evaluations.

        Uses the mean of past experiment costs when ``cost_column`` is
        available, otherwise returns ``cost_value``.
        """
        if (
            self.cost_column is not None
            and self.cost_column in experiments.columns
        ):
            costs = experiments[self.cost_column].dropna()
            if len(costs) > 0:
                return float(costs.mean())
        return self.cost_value

    def _effective_cost_callable(
        self, cost_estimate: float
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a cost callable suitable for ``_LogEIPCWithCost``.

        If ``cost_callable`` was supplied by the user, return it directly.
        Otherwise wrap the scalar ``cost_estimate`` as a constant callable.
        """
        if self.cost_callable is not None:
            return self.cost_callable
        return lambda X: torch.full(
            (X.shape[0],), cost_estimate, dtype=X.dtype, device=X.device
        )

    def _compute_max_log_eipc(
        self,
        model: Model,
        best_f: float,
        bounds_lower: torch.Tensor,
        bounds_upper: torch.Tensor,
        cost_estimate: float,
    ) -> float:
        """Return max LogEIPC over the domain by random sampling.

        Delegates per-batch evaluation to ``compute_log_eipc_at``.
        """
        n_dims = bounds_lower.shape[0]
        X_random = bounds_lower + (bounds_upper - bounds_lower) * torch.rand(
            self.n_samples, n_dims, **tkwargs
        )

        max_log_eipc = float("-inf")
        for start in range(0, self.n_samples, self.batch_size):
            X_batch = X_random[start : start + self.batch_size]
            batch_vals = self.compute_log_eipc_at(
                model, X_batch, best_f, cost_estimate
            )
            batch_max = float(batch_vals.max())
            if batch_max > max_log_eipc:
                max_log_eipc = batch_max
        return max_log_eipc

    def _optimize_max_log_eipc(
        self,
        strategy,
        experiments: pd.DataFrame,
        best_f: float,
        cost_estimate: float,
    ) -> float:
        """Return max LogEIPC via BoFire's acquisition function optimizer.

        Uses ``_LogEIPCWithCost`` so that input-dependent cost callables are
        correctly accounted for during optimisation.
        """
        # LogExpectedImprovementPerCost computes LogEI - alpha*log(c(x)).
        # lambda_cost is a constant shift that doesn't affect the argmax,
        # so we apply it only when reading back the value.
        acqf = LogExpectedImprovementPerCost(
            model=strategy.model,
            best_f=best_f,
            cost_callable=self._effective_cost_callable(cost_estimate),
            alpha=self.alpha,
        )
        candidates = strategy.acqf_optimizer.optimize(
            candidate_count=1,
            acqfs=[acqf],
            domain=strategy.domain,
            experiments=experiments,
        )
        transformed = strategy.domain.inputs.transform(
            candidates, strategy.input_preprocessing_specs
        )
        X_best = torch.from_numpy(transformed.values).to(**tkwargs)
        with torch.no_grad():
            val = acqf(X_best.unsqueeze(-2))
        return float(val.item()) - float(np.log(self.lambda_cost))

    def compute_log_eipc_at(
        self,
        model: Model,
        X: torch.Tensor,
        best_f: float,
        cost_estimate: float,
    ) -> np.ndarray:
        """Evaluate LogEIPC at given points. Useful for plotting over a dense grid.

        When ``cost_callable`` is set, costs are evaluated per-point; otherwise
        the scalar ``cost_estimate`` is used.

        Args:
            model: Fitted GP model (e.g. ``strategy.model``).
            X: Candidate points shaped ``(n, d)`` in transformed input space.
            best_f: Best observed objective value (original scale).
            cost_estimate: Scalar cost fallback (used when ``cost_callable`` is None).

        Returns:
            Array of shape ``(n,)`` with LogEIPC values.
        """
        logei_acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
        with torch.no_grad():
            log_ei = logei_acqf(X.unsqueeze(-2))           # (n,)
            costs = self._effective_cost_callable(cost_estimate)(X)  # (n,)
            log_eipc = (
                log_ei
                - self.alpha * torch.log(costs.clamp(min=1e-12))
                - float(np.log(self.lambda_cost))
            )
        return log_eipc.numpy()

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Return LogEIPC metrics, or an empty dict when not applicable."""
        if not strategy.is_fitted or strategy.model is None:
            return {}
        if len(experiments) < 2:
            return {}
        if strategy.model.num_outputs != 1:
            return {}

        output_key = strategy.domain.outputs.get_keys()[0]
        best_f = float(experiments[output_key].min())

        cost_estimate = self._get_cost_estimate(experiments)
        if cost_estimate <= 0:
            return {}

        # When cost_model="gp" and a cost_column is available, replace the
        # scalar mean with a GP-fitted per-point cost callable for this call.
        # The user-supplied cost_callable always takes priority.
        _saved_callable = self.cost_callable
        if self.cost_model == "gp" and self.cost_callable is None:
            fitted = self._fit_cost_gp(strategy, experiments)
            if fitted is not None:
                self.cost_callable = fitted

        if self.search_method == "optimize":
            max_log_eipc = self._optimize_max_log_eipc(
                strategy, experiments, best_f, cost_estimate
            )
        else:
            bounds = strategy.domain.inputs.get_bounds(
                specs=strategy.input_preprocessing_specs
            )
            lower = torch.tensor(bounds[0], **tkwargs)
            upper = torch.tensor(bounds[1], **tkwargs)
            max_log_eipc = self._compute_max_log_eipc(
                strategy.model, best_f, lower, upper, cost_estimate
            )

        self.cost_callable = _saved_callable  # restore after GP override

        return {
            "max_log_eipc": max_log_eipc,
            "best_f": best_f,
            "cost_estimate": cost_estimate,
            "lambda_cost": self.lambda_cost,
        }
