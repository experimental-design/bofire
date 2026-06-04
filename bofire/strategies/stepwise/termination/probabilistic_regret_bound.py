"""Probabilistic regret bound stopping criterion (Wilson, 2024)."""

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.model import Model
from botorch.sampling.pathwise import draw_matheron_paths
from scipy.optimize import minimize as scipy_minimize

from bofire.strategies.stepwise.termination.evaluator import TerminationEvaluator
from bofire.strategies.stepwise.termination.utils import clopper_pearson_ci
from bofire.utils.torch_tools import tkwargs


def _minimize_sample_paths(
    paths,
    bounds_lower: torch.Tensor,
    bounds_upper: torch.Tensor,
    n_random: int = 512,
    n_starts: int = 8,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    ftol: float = 1e-9,
    sign: float = 1.0,
) -> np.ndarray:
    """Find the minimum value of ``sign * f`` for each GP posterior sample path.

    With ``sign = +1`` this is the per-path minimum (minimisation); with
    ``sign = -1`` it is the negated per-path maximum (maximisation), so the
    returned value is always the optimum in the "minimise ``g = sign*f``" frame.

    Uses a two-phase approach:

    1. **Random search** — evaluate all paths at ``n_random`` uniformly drawn
       domain points to identify the ``n_starts`` most promising starting
       points per trajectory (lowest ``sign * f``).
    2. **Local refinement** — run ``method`` from each of those starts.

    Args:
        paths: BoTorch ``MatheronPath`` callable.  ``paths(X)`` with
            ``X`` shaped ``[m, d]`` returns ``[n_samples, m]``.
        bounds_lower: Lower bounds of the (transformed) input domain, ``[d]``.
        bounds_upper: Upper bounds, ``[d]``.
        n_random: Number of random domain points for start selection.
        n_starts: Number of local-optimisation starts per trajectory.
        method: scipy optimisation method.  Default ``"L-BFGS-B"`` (best for
            smooth box-bounded problems with gradient information).  Other
            gradient-compatible bounded methods: ``"TNC"``, ``"SLSQP"``.
        maxiter: Maximum number of optimiser iterations per start.
        ftol: Absolute function-value tolerance for convergence.
        sign: ``+1`` to minimise the paths, ``-1`` to maximise them.

    Returns:
        Array of shape ``[n_samples]`` with ``min_x sign*f^i(x)`` per path.
    """
    d = bounds_lower.shape[0]
    dtype = bounds_lower.dtype
    bounds_np = list(zip(bounds_lower.tolist(), bounds_upper.tolist()))

    # Phase 1: batch-evaluate all paths at n_random random points
    X_rand = bounds_lower + (bounds_upper - bounds_lower) * torch.rand(
        n_random, d, dtype=dtype
    )
    with torch.no_grad():
        vals_rand = sign * paths(X_rand)  # [n_samples, n_random], in g-space
    n_samples = vals_rand.shape[0]

    n_top = min(n_starts, n_random)
    top_idx = vals_rand.argsort(dim=1)[:, :n_top]  # [n_samples, n_top]

    # Phase 2: L-BFGS-B from the n_top best starts per trajectory
    minima = np.full(n_samples, np.inf)

    for i in range(n_samples):
        best_val = np.inf
        for s in range(n_top):
            x0 = X_rand[top_idx[i, s]].detach().numpy().copy()

            def obj(x_np, _i=i):
                x_t = torch.tensor(x_np, dtype=dtype).unsqueeze(0)
                x_t.requires_grad_(True)
                val = sign * paths(x_t)[_i].squeeze()
                grad = torch.autograd.grad(val, x_t)[0]
                return float(val.item()), grad.squeeze(0).detach().numpy().copy()

            try:
                res = scipy_minimize(
                    obj,
                    x0,
                    method=method,
                    jac=True,
                    bounds=bounds_np,
                    options={"maxiter": maxiter, "ftol": ftol},
                )
                if res.fun < best_val:
                    best_val = float(res.fun)
            except Exception:
                v0 = float(vals_rand[i, top_idx[i, s]].item())
                if v0 < best_val:
                    best_val = v0

        minima[i] = best_val

    return minima


def _run_prb_level_test(
    sampler_fn: Callable[[int], np.ndarray],
    level: float,
    delta_est: float,
    n_samples_max: int,
    n_test: int,
    initial_batch: int = 16,
    batch_growth: float = 1.5,
) -> tuple:
    """Sequential Clopper-Pearson level test for the PRB stopping criterion.

    Tests whether ``P(regret > ε) ≤ level`` for any test point, with total
    false-positive risk bounded by ``delta_est``.

    Convergence follows the ``ANY_LE`` rule from Wilson (2024): the loop
    stops as soon as any test point's CI upper bound drops below ``level``
    (criterion met for that point), OR all test points' CI lower bounds
    exceed ``level`` (criterion definitely not met for any point).

    Args:
        sampler_fn: ``(n_batch) → int[n_test, n_batch]``.  Each call draws
            ``n_batch`` independent GP posterior trajectories and returns
            indicator samples where ``1`` means the regret of test point
            ``j`` under trajectory ``i`` exceeded ``ε``.
        level: ``delta_mod``.  A test point satisfies the criterion
            when its estimated ``P(indicator = 1) ≤ delta_mod``.
        delta_est: Total risk budget.  Per-step risk follows the power
            schedule ``(0.1 / 1.1) × delta_est × t^{−1.1}``, which sums to
            ``≤ delta_est``.  Bonferroni-corrected over ``n_test`` points.
        n_samples_max: Maximum total indicator samples before the loop exits.
        n_test: Number of test points evaluated in parallel.
        initial_batch: Initial cumulative sample target N (= first batch size).
        batch_growth: Geometric growth factor β.  Cumulative targets are
            N, βN, β²N, … so per-step batches are N, (β−1)N, β(β−1)N, …

    Returns:
        ``(estimates, converged_below, total_n, cis)`` where

        - ``estimates`` — ``float[n_test]``, ``P̂(regret > ε)`` per point.
        - ``converged_below`` — ``bool[n_test]``, True when CI upper < level.
        - ``total_n`` — total indicator samples drawn.
        - ``cis`` — ``float[n_test, 2]``, final ``(lower, upper)`` CIs for
          ``P(regret > ε)``.
    """
    total_n = 0
    total_k = np.zeros(n_test, dtype=np.int64)
    step = 0

    lowers = np.zeros(n_test)
    uppers = np.ones(n_test)
    converged_below = np.zeros(n_test, dtype=bool)

    while total_n < n_samples_max:
        step += 1
        # Match trieste_stopping: n_j is the *cumulative* target at step j.
        # The batch drawn is the increment n_j − n_{j−1}, not n_j itself.
        # This makes the cumulative sample counts geometric (16, 24, 36, …)
        # while the per-step batches are their differences (16, 8, 12, 18, …).
        n_target = int(initial_batch * batch_growth ** (step - 1))
        n_batch = min(n_target - total_n, n_samples_max - total_n)
        if n_batch <= 0:
            continue  # integer rounding made this step a no-op; advance to next

        indicators = sampler_fn(n_batch)  # [n_test, n_batch]
        total_n += n_batch
        total_k += indicators.sum(axis=1)

        # Power-law risk schedule; Bonferroni correction over test points.
        step_risk = (0.1 / 1.1) * delta_est * (float(step) ** -1.1)
        per_point_risk = step_risk / max(n_test, 1)

        any_satisfied = False
        all_confirmed_above = True

        for j in range(n_test):
            lower, upper = clopper_pearson_ci(int(total_k[j]), total_n, per_point_risk)
            lowers[j] = lower
            uppers[j] = upper
            if upper < level:
                converged_below[j] = True
                any_satisfied = True
            if lower <= level:
                all_confirmed_above = False

        if any_satisfied or all_confirmed_above:
            break

    estimates = total_k.astype(float) / total_n
    cis = np.stack([lowers, uppers], axis=1)
    return estimates, converged_below, total_n, cis


class ProbabilisticRegretBoundEvaluator(TerminationEvaluator):
    """Evaluator for the probabilistic regret bound (PRB) stopping criterion.

    Terminates BO when the model-based probability that the incumbent's simple
    regret is at most ε is certified to exceed ``1 − delta_mod`` via a
    Clopper-Pearson sequential hypothesis test over GP posterior sample paths.

    Following the paper's notation (Wilson, 2024), the total risk is
    ``δ = delta_mod + delta_est``, where:

    - ``delta_mod`` (δ_mod): the probability that the stopping condition
      triggers a false positive under the model.  The CP test checks
      ``P̂(regret > ε) ≤ delta_mod``.
    - ``delta_est`` (δ_est): the false-positive risk allocated to the
      Clopper-Pearson estimation error.

    The criterion is satisfied when ``P̂(regret > ε) ≤ delta_mod``, i.e. the
    model-based probability that the incumbent has regret exceeding ε is
    certified (by the CP test) to be at most delta_mod.

    At each call, the evaluator:

    1. Draws batches of GP posterior paths via BoTorch's Matheron-rule sampler.
    2. Globally minimises each path via multistart L-BFGS-B to obtain per-path
       optima ``f^i_*``.
    3. Computes regret indicators ``𝟙(f^i(x) − f^i_* > ε)`` for each test
       point ``x`` and path ``i``.
    4. Updates Clopper-Pearson confidence intervals and stops when the CI
       conclusively excludes ``delta_mod``, or ``n_samples_max`` is exhausted.

    Requires a BoTorch ``SingleTaskGP``-compatible model (RBF, Matérn, and
    most stationary kernels).  Single-output minimisation only.

    Args:
        epsilon: Absolute simple regret threshold in original Y units.  If
            ``None`` (default), computed as ``epsilon_relative × (y_max − y_min)``.
        epsilon_relative: Fractional ε relative to the observed Y range.
            Default ``0.01`` (1 %).  Ignored when ``epsilon`` is set.
        delta_mod: Model-based risk budget δ_mod.  Stop when
            ``P̂(regret > ε) ≤ delta_mod``.  Default ``0.05``.
            Together with ``delta_est`` this gives total risk δ = 0.10.
            Use ``delta_mod = 0.025`` and ``delta_est = 0.025`` to match the
            paper's experiments (total δ = 0.05).
        delta_est: Estimation-error risk budget δ_est for the CP test.
            Default ``0.05``.
        enforce_convergence: If ``True`` (default), only declare the criterion
            satisfied when the CP CI conclusively excludes the level.  If
            ``False``, uses the raw MC probability estimate.
        n_samples_max: Maximum total GP path samples per BO step.
            Default ``1024``.
        initial_batch: First CP test batch size.  Default ``16``.
        batch_growth: Geometric growth factor for batch sizes.  Default ``1.5``
            (→ batches of 16, 24, 36, 54, …).
        n_starts: Local-optimisation starts per path.  Default ``8``.
        n_random: Random domain points for identifying promising start
            candidates.  Default ``512``.
        optim_method: scipy optimisation method passed to
            ``_minimize_sample_paths``.  Default ``"L-BFGS-B"``.
        optim_maxiter: Maximum iterations per start.  Default ``200``.
        optim_ftol: Function-value convergence tolerance.  Default ``1e-9``.
        n_test_points: Candidate points to evaluate the criterion at.  ``1``
            (default) tests the incumbent only; values ``> 1`` also include
            the ``n_test_points − 1`` in-sample points with the lowest GP
            posterior mean.

    References:
        Wilson (2024): "Stopping Bayesian Optimization with Probabilistic
            Regret Bounds" (NeurIPS 2024).
            Reference code: https://github.com/j-wilson/trieste_stopping.
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        epsilon_relative: float = 0.01,
        delta_mod: float = 0.05,
        delta_est: float = 0.05,
        enforce_convergence: bool = True,
        n_samples_max: int = 1024,
        initial_batch: int = 16,
        batch_growth: float = 1.5,
        n_starts: int = 8,
        n_random: int = 512,
        n_test_points: int = 1,
        optim_method: str = "L-BFGS-B",
        optim_maxiter: int = 200,
        optim_ftol: float = 1e-9,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.delta_mod = delta_mod
        self.delta_est = delta_est
        self.enforce_convergence = enforce_convergence
        self.n_samples_max = n_samples_max
        self.initial_batch = initial_batch
        self.batch_growth = batch_growth
        self.n_starts = n_starts
        self.n_random = n_random
        self.n_test_points = n_test_points
        self.optim_method = optim_method
        self.optim_maxiter = optim_maxiter
        self.optim_ftol = optim_ftol

    def _get_epsilon(self, y_vals: np.ndarray) -> float:
        """Return the ε threshold in original Y units."""
        if self.epsilon is not None:
            return self.epsilon
        y_range = float(y_vals.max() - y_vals.min())
        return max(self.epsilon_relative * y_range, 1e-8)

    def _get_test_points(
        self,
        model: Model,
        X_all: torch.Tensor,
        incumbent_pos: int,
        sign: float = 1.0,
    ) -> torch.Tensor:
        """Return test points in transformed input space.

        The incumbent is always first.  Additional slots are filled with the
        in-sample points that are best under the model — lowest ``sign * mu``
        (lowest posterior mean for minimisation, highest for maximisation).
        """
        x_incumbent = X_all[[incumbent_pos]]  # [1, d]
        if self.n_test_points == 1:
            return x_incumbent

        with torch.no_grad():
            means = sign * model.posterior(X_all).mean.squeeze(-1)  # [n], g-space

        sorted_pos = means.argsort().tolist()
        extras = [X_all[[pos]] for pos in sorted_pos if pos != incumbent_pos][
            : self.n_test_points - 1
        ]

        return torch.cat([x_incumbent] + extras, dim=0)  # [n_test, d]

    def _evaluate_core(
        self,
        model: Model,
        X_test: torch.Tensor,
        bounds_lower: torch.Tensor,
        bounds_upper: torch.Tensor,
        epsilon: float,
        sign: float = 1.0,
    ) -> Dict[str, Any]:
        """Core evaluation logic called by ``evaluate``.

        Args:
            model: Fitted BoTorch GP model.
            X_test: Test points shaped ``[n_test, d]`` (already transformed).
            bounds_lower: Lower bounds of the (transformed) domain, ``[d]``.
            bounds_upper: Upper bounds, ``[d]``.
            epsilon: Absolute regret threshold in original Y units.
            sign: ``+1`` for minimisation, ``-1`` for maximisation.  Regret is
                computed in the "minimise ``g = sign*f``" frame, so for
                maximisation paths are maximised and the regret is
                ``g(x) - min_g = sign*f(x) - min_x sign*f(x) = max f - f(x)``.

        Returns:
            Metrics dict, or empty dict on failure.
        """
        n_test = X_test.shape[0]
        level = self.delta_mod

        def sampler_fn(n_batch: int) -> np.ndarray:
            # Any failure here (path sampling / minimisation) is allowed to
            # propagate to the outer handler, which returns an empty metrics
            # dict — the condition then keeps optimizing.  This is deliberately
            # not caught and replaced with random indicators, which would
            # corrupt the Clopper-Pearson estimate.
            with torch.no_grad():
                paths = draw_matheron_paths(model, sample_shape=torch.Size([n_batch]))
            minima = _minimize_sample_paths(
                paths,
                bounds_lower,
                bounds_upper,
                self.n_random,
                self.n_starts,
                self.optim_method,
                self.optim_maxiter,
                self.optim_ftol,
                sign,
            )
            with torch.no_grad():
                path_vals = sign * paths(X_test).numpy()  # [n_batch, n_test], g-space
            regrets = path_vals - minima[:, np.newaxis]  # [n_batch, n_test], ≥ 0
            return (regrets > epsilon).astype(np.int64).T  # [n_test, n_batch]

        try:
            estimates, converged_below, n_used, cis = _run_prb_level_test(
                sampler_fn=sampler_fn,
                level=level,
                delta_est=self.delta_est,
                n_samples_max=self.n_samples_max,
                n_test=n_test,
                initial_batch=self.initial_batch,
                batch_growth=self.batch_growth,
            )
        except Exception:
            return {}

        best_j = int(np.argmin(estimates))
        prob_regret_ok = float(1.0 - estimates[best_j])
        ci_lower = float(1.0 - cis[best_j, 1])
        ci_upper = float(1.0 - cis[best_j, 0])

        if self.enforce_convergence:
            criterion_satisfied = bool(converged_below[best_j])
        else:
            criterion_satisfied = bool(1.0 - prob_regret_ok <= self.delta_mod)

        return {
            "prob_regret_ok": prob_regret_ok,
            "epsilon": epsilon,
            "delta_mod": self.delta_mod,
            "criterion_satisfied": criterion_satisfied,
            "n_samples_used": n_used,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "converged": bool(converged_below[best_j]),
        }

    def evaluate(
        self,
        strategy,
        experiments: pd.DataFrame,
        iteration: int,
    ) -> Dict[str, Any]:
        """Return PRB metrics, or an empty dict when not applicable.

        Returns a dict with keys:

        - ``prob_regret_ok`` — ``P̂(regret ≤ ε)`` for the best test point.
        - ``epsilon`` — the ε value used (original Y units).
        - ``delta_mod`` — the configured model-risk threshold.
        - ``criterion_satisfied`` — ``True`` when stopping is recommended.
        - ``n_samples_used`` — total GP path samples drawn.
        - ``ci_lower``, ``ci_upper`` — Clopper-Pearson CI bounds on
          ``P(regret ≤ ε)`` at coverage ``1 − per_point_risk`` (derived from
          ``delta_est`` via the per-step risk schedule), not a fixed 95 %.
        - ``converged`` — whether the CP CI conclusively excluded the level.
        """
        if not strategy.is_fitted or strategy.model is None:
            return {}
        if len(experiments) < 2:
            return {}
        if strategy.model.num_outputs != 1:
            return {}
        sign = self._objective_sign(strategy)
        if sign is None:
            return {}

        model = strategy.model
        input_keys = strategy.domain.inputs.get_keys()
        output_key = strategy.domain.outputs.get_keys()[0]

        y_vals = experiments[output_key].values.astype(float)
        if len(y_vals) < 2:
            return {}
        epsilon = self._get_epsilon(y_vals)

        transformed = strategy.domain.inputs.transform(
            experiments[input_keys], strategy.input_preprocessing_specs
        )
        X_all = torch.from_numpy(transformed.values).to(**tkwargs)
        # Incumbent is the best point in the "minimise sign*y" frame.
        incumbent_pos = int(np.argmin(sign * y_vals))

        try:
            X_test = self._get_test_points(model, X_all, incumbent_pos, sign)
        except Exception:
            return {}

        bounds = strategy.domain.inputs.get_bounds(
            specs=strategy.input_preprocessing_specs
        )
        lower = torch.tensor(bounds[0], **tkwargs)
        upper = torch.tensor(bounds[1], **tkwargs)

        return self._evaluate_core(model, X_test, lower, upper, epsilon, sign)
