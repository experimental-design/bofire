"""UCBLCBRegretEvaluator stopping criterion evaluator."""

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch

from bofire.strategies.stepwise.termination.evaluator import RegretBoundEvaluator
from bofire.utils.torch_tools import tkwargs


class UCBLCBRegretEvaluator(RegretBoundEvaluator):
    """Evaluator for the UCB-LCB regret bound based on the approach from Makarova et al., 2022.

    The bound is ``min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)``
    using the GP-UCB formulation (Srinivas et al., 2010).  Stopping is
    triggered when this gap is small — meaning the best candidate already
    evaluated is nearly as good as the best anywhere in the domain.

    Single-output only.

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
        lcb_method: How the domain-wide minimum LCB is found.
            ``"sample"`` (default) draws ``n_samples_lcb`` random points —
            matches the reference implementation.  ``"optimize"`` uses
            BoFire's acquisition function optimizer for higher accuracy.
        n_samples_lcb: Number of random domain points used when
            ``lcb_method="sample"`` to approximate the minimum LCB over the
            domain. Default ``2000``.
        batch_size: If set, chunk GP posterior evaluation during sampling into
            batches of this size to bound memory.  ``None`` (default) evaluates
            all points in a single posterior call.
        topq: Fraction of best observations used for the regret-bound GP.
            Makarova et al. (2022) found fitting the bound on the best ~50 %
            of observations works best.  ``1.0`` (default) uses all
            observations; values below 1 refit a copy of the strategy on the
            best fraction (the main strategy's GP is unaffected).  Only
            engages once more than ``min_topq`` observations are available.
        min_topq: Minimum observations kept under top-q filtering. Default ``20``.

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
        lcb_method: Literal["sample", "optimize"] = "sample",
        n_samples_lcb: int = 2000,
        batch_size: Optional[int] = None,
        topq: float = 1.0,
        min_topq: int = 20,
    ):
        super().__init__(
            delta=delta,
            beta_scale=beta_scale,
            beta_log_multiplier=beta_log_multiplier,
            beta_log_denominator=beta_log_denominator,
            beta_min=beta_min,
            beta_t_offset=beta_t_offset,
            lcb_method=lcb_method,
        )
        self.fallback_noise_variance = fallback_noise_variance
        self.n_samples_lcb = n_samples_lcb
        self.batch_size = batch_size
        self.topq = topq
        self.min_topq = min_topq

    def _apply_topq(
        self,
        strategy,
        experiments: pd.DataFrame,
        sign: float,
    ) -> Optional[tuple]:
        """Refit a copy of the strategy on the best ``topq`` fraction.

        Returns ``(eval_strategy, eval_experiments)``, or ``None`` when the
        refit fails (callers should skip evaluation).  With ``topq=1.0`` or
        too few observations, returns the inputs unchanged.
        """
        if self.topq >= 1.0:
            return strategy, experiments
        output_key = strategy.domain.outputs.get_keys()[0]
        y_values = experiments[output_key].values
        n = len(y_values)
        topn = max(self.min_topq, int(n * self.topq))
        if topn >= n:
            return strategy, experiments
        # Best = lowest ``sign * y`` (lowest y for minimisation, highest for
        # maximisation).
        top_indices = np.argsort(sign * y_values)[:topn]
        eval_experiments = experiments.iloc[top_indices].reset_index(drop=True)

        from bofire.strategies.mapper import map as map_strategy

        try:
            eval_strategy = map_strategy(strategy._data_model)
            eval_strategy.tell(eval_experiments)
        except Exception:
            return None
        return eval_strategy, eval_experiments

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
        direction = self._objective_sign(strategy)
        if direction is None:
            return {}
        sign = -direction  # minimisation frame: +1 minimise / -1 maximise

        # Top-q: compute the bound from a copy refit on the best fraction.
        filtered = self._apply_topq(strategy, experiments, sign)
        if filtered is None:
            return {}
        strategy, experiments = filtered

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
            strategy.model,
            X_evaluated,
            sqrt_beta,
            lower,
            upper,
            self.n_samples_lcb,
            self.batch_size,
            strategy=strategy,
            experiments=experiments,
            sign=sign,
        )

        # Likelihood noise for the threshold, un-standardized to match the
        # original-scale regret bound (Standardize learns noise in std space).
        try:
            estimated_noise_var = (
                strategy.model.likelihood.noise.item()
                * self.get_output_scale(strategy.model) ** 2
            )
        except Exception:
            estimated_noise_var = self.fallback_noise_variance

        return {
            "regret_bound": regret_bound,
            "min_ucb_evaluated": min_ucb_evaluated,
            "min_lcb_domain": min_lcb_domain,
            "estimated_noise_variance": estimated_noise_var,
            "beta": beta,
        }
