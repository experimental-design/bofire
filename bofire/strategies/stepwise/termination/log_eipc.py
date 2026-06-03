"""LogEIPCEvaluator stopping criterion evaluator."""

from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    PosteriorTransform,
    _log_ei_helper,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.strategies.stepwise.termination.evaluator import TerminationEvaluator
from bofire.utils.torch_tools import tkwargs


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


class LogEIPCEvaluator(TerminationEvaluator):
    """Cost-aware stopping criterion based on Xie et al., 2025.

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
        if self.cost_column is not None and self.cost_column in experiments.columns:
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
            batch_vals = self.compute_log_eipc_at(model, X_batch, best_f, cost_estimate)
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
            log_ei = logei_acqf(X.unsqueeze(-2))  # (n,)
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
