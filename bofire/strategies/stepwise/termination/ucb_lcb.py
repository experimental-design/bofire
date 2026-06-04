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
        sign = self._objective_sign(strategy)
        if sign is None:
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
