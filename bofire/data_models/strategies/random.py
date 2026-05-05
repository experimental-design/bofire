from typing import Annotated, Literal, Optional, Type

from pydantic import Field

from bofire.data_models.constraints.api import (
    CategoricalExcludeConstraint,
    Constraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    """Strategy for drawing random samples from a domain subject to its constraints.

    Sampling proceeds in four regimes, picked automatically based on the
    constraints present in the domain:

    1. **Unconstrained / categorical-only domains.** Samples are drawn directly
       from each input feature using ``fallback_sampling_method`` (uniform,
       Sobol, or LHS).
    2. **Linear/interpoint-constrained domains.** A hit-and-run polytope sampler
       (``botorch.sample_q_batches_from_polytope``) draws candidates that
       satisfy linear equality, linear inequality, and interpoint equality
       constraints. Categorical and discrete inputs are sampled independently
       with the fallback method.
    3. **NChooseK and/or ``allow_zero`` features.** The strategy draws up to
       ``max_combinations`` distinct active-feature subsets uniformly from all
       valid subsets (one group per ``NChooseKConstraint``, plus one singleton
       group per ``ContinuousInput`` with ``allow_zero=True`` outside any
       NChooseK). For each drawn subset the unselected zeroable features are
       pinned to ``0`` and the remaining features are sampled via the polytope
       sampler. Final candidates are concatenated and uniformly subsampled.
    4. **Nonlinear or product constraints.** When constraints are present that
       cannot be enforced directly by the polytope sampler, regimes 1-3 are
       used as a proposal distribution and rejection sampling filters
       candidates until ``candidate_count`` valid samples are found.

    Attributes:
        fallback_sampling_method: Sampling method for unconstrained / fixed
            inputs and for categorical and discrete features.
        n_burnin: Burn-in length for the hit-and-run polytope sampler.
        n_thinning: Thinning factor for the hit-and-run polytope sampler.
        num_base_samples: Batch size used when drawing proposals during
            rejection sampling. If ``None``, the requested ``candidate_count``
            is used.
        max_iters: Maximum number of rejection-sampling iterations before the
            strategy gives up. Each iteration draws ``num_base_samples``
            candidates.
        max_combinations: Maximum number of distinct active-feature subsets to
            draw per ``ask`` when NChooseK or ``allow_zero`` features are
            present. Larger values give a better mix of subsets at the cost of
            more polytope-sampler invocations.
        nchoosek_max_iters: Maximum number of rejection-sampling attempts when
            drawing a single valid active-feature subset under overlapping
            ``NChooseKConstraint``s. Independent from ``max_iters``.
        sampler_kwargs: Extra keyword arguments forwarded to the fallback
            sampler (e.g. ``{"scramble": True}`` for Sobol).
    """

    type: Literal["RandomStrategy"] = "RandomStrategy"
    fallback_sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM
    n_burnin: Annotated[int, Field(ge=1)] = 1000
    n_thinning: Annotated[int, Field(ge=1)] = 32
    num_base_samples: Optional[Annotated[int, Field(gt=0)]] = None
    max_iters: Annotated[int, Field(gt=0)] = 1000
    max_combinations: Annotated[int, Field(gt=0)] = 64
    nchoosek_max_iters: Annotated[int, Field(gt=0)] = 1000
    sampler_kwargs: Optional[dict] = None

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            LinearEqualityConstraint,
            NChooseKConstraint,
            InterpointEqualityConstraint,
            NonlinearInequalityConstraint,
            ProductInequalityConstraint,
            CategoricalExcludeConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return True
