"""ExpMinRegretGapEvaluator stopping criterion evaluator."""

import base64
import io
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.model import Model
from scipy.stats import norm

from bofire.strategies.stepwise.termination.evaluator import RegretBoundEvaluator
from bofire.utils.torch_tools import tkwargs


class ExpMinRegretGapEvaluator(RegretBoundEvaluator):
    """Evaluator for the expected minimum regret gap from Ishibashi et al., 2023.

    Computes a stopping value that upper-bounds the change in expected minimum
    simple regret between consecutive BO iterations:

        value_t = delta_f + ei_diff + kappa * sqrt(KL / 2)

    where ``delta_f`` is the change in the GP mean at the incumbent,
    ``ei_diff`` is the expected improvement from switching incumbents,
    ``kappa`` is the UCB-LCB regret bound from the previous model, and
    ``KL`` is the KL divergence of the old GP prior vs. the updated
    posterior at the newly observed point.

    Stateful: stores the previous GP model and incumbent index between calls.
    The first call always returns an empty dict. Single-output only.

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
        sign: float = 1.0,
    ) -> float:
        """Expected improvement from switching incumbents.

        If the incumbent changed, computes ``E[max(g(x*_new) - g(x*_old), 0)]``
        under the new GP's joint posterior at the two incumbent locations, in
        the "minimise ``g = sign*f``" frame (``sign`` flips the improvement
        direction for maximisation).  The difference's variance is unchanged
        by ``sign``.
        """
        X_pair = torch.cat([X_new_incumbent, X_old_incumbent], dim=0)
        with torch.no_grad():
            posterior = model.posterior(X_pair)
            mu = posterior.mean.squeeze(-1)
            cov = posterior.distribution.covariance_matrix

        g = sign * float((mu[0] - mu[1]).item())
        var_diff = float((cov[0, 0] - 2 * cov[0, 1] + cov[1, 1]).item())

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
        direction = self._objective_sign(strategy)
        if direction is None:
            return {}
        sign = -direction  # minimisation frame: +1 minimise / -1 maximise

        input_keys = strategy.domain.inputs.get_keys()
        output_key = strategy.domain.outputs.get_keys()[0]
        dimensionality = len(input_keys)
        n_experiments = len(experiments)

        # Incumbent is the best point in the "minimise sign*y" frame:
        # argmin(y) for minimisation, argmax(y) for maximisation.
        incumbent_idx = int((sign * experiments[output_key]).idxmin())

        # First call: save state and return empty.
        if self._prev_model is None or n_experiments <= self._prev_n_experiments:
            self._save_state(strategy, experiments, incumbent_idx)
            return {}

        # Assume one new point added since the last call; take the last.
        new_point_idx = n_experiments - 1
        y_new = float(experiments[output_key].iloc[new_point_idx])

        preprocessing_specs = strategy.input_preprocessing_specs
        all_inputs = experiments[input_keys]
        transformed = strategy.domain.inputs.transform(all_inputs, preprocessing_specs)
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
        bounds = strategy.domain.inputs.get_bounds(specs=preprocessing_specs)
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        kappa, _, _ = self._ucb_lcb_regret_bound(
            self._prev_model,
            X_prev,
            sqrt_beta,
            lower,
            upper,
            self.n_samples_lcb,
            self.batch_size,
            strategy=strategy,
            experiments=experiments,
            sign=sign,
        )

        # ei_diff: expected improvement from switching incumbents.
        if incumbent_idx == self._prev_incumbent_idx:
            ei_diff = 0.0
        else:
            ei_diff = self._compute_ei_diff(
                strategy.model, x_incumbent_new, x_incumbent_old, sign=sign
            )

        stopping_value = delta_f + ei_diff + kappa * np.sqrt(0.5 * kl)

        threshold_adaptive = None
        threshold_median = None

        if self.threshold_mode in ("adaptive", "adaptive_median"):
            # Adaptive threshold in raw scale (reference uses normalize_Y=False).
            threshold_adaptive = self._compute_threshold_adaptive(
                self._prev_model,
                x_incumbent_new,
                old_var,
                noise_var_original,
                kappa,
            )

        self._seq_values.append(stopping_value)
        if self.threshold_mode in ("median", "adaptive_median"):
            threshold_median = self._compute_threshold_median(
                self._seq_values,
                self.start_timing,
                self.rate,
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
        self._prev_input_preprocessing_specs = state["prev_input_preprocessing_specs"]
        self._seq_values = state["seq_values"]
