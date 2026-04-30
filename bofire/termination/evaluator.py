"""Termination evaluators that compute metrics for termination conditions."""

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from scipy.stats import norm

from bofire.utils.torch_tools import tkwargs


class TerminationEvaluator(ABC):
    """Base class for termination evaluators.

    Computes metrics from a BO strategy that termination conditions use to
    decide whether to stop.
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
        with torch.no_grad():
            posterior = model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.squeeze(-1).sqrt()
        return float((mean + sqrt_beta * std).min().item())

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
        neg_lcb = _NegLowerConfidenceBound(
            model=strategy.model, sqrt_beta=float(sqrt_beta)
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


class _NegLowerConfidenceBound(AnalyticAcquisitionFunction):
    """Negative LCB, so that maximising it minimises LCB."""

    def __init__(self, model: Model, sqrt_beta: float):
        super().__init__(model=model)
        self.sqrt_beta = sqrt_beta

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X.squeeze(-2))
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).sqrt()
        lcb = mean - self.sqrt_beta * std
        return -lcb  # negate so maximize -> minimize LCB


class UCBLCBRegretEvaluator(TerminationEvaluator):
    """Evaluator for the UCB-LCB regret bound (Makarova et al., 2022).

    The bound is ``min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)``
    using the GP-UCB formulation (Srinivas et al., 2010). ``lcb_method``
    selects between random sampling (``"sample"``, default — matching the
    reference implementation at github.com/amazon-science/bo-early-stopping)
    and BoTorch acquisition optimization (``"optimize"``) for the domain-wide
    min LCB.

    Single-output minimisation only.

    Reference:
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
    The first call returns an empty dict. Single-output minimisation only.

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
