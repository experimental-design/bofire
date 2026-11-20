from typing import Literal, Optional, Type

from pydantic import Field, PositiveInt

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
from bofire.data_models.features.api import Feature
from bofire.data_models.objectives.api import Objective
from bofire.data_models.samplers.api import AnySampler, UniformSampler
from bofire.data_models.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    """Draws random candidates that satisfy the constraints of the domain.

    Use it for initial designs before a model-based strategy takes over. How the
    candidates are drawn depends on the constraints:

    1. Without constraints, every input is sampled by `fallback_sampler`.
    2. With linear or interpoint constraints, the continuous inputs are
       drawn by a random walk inside the feasible region (hit-and-run); categorical
       and discrete inputs are still sampled by `fallback_sampler`.
    3. With NChooseK constraints or inputs that may be zero (`allow_zero`), the
       strategy first draws up to `max_combinations` sets of active inputs, sets the
       other ones to zero and samples each set as in 2.
    4. Nonlinear and product constraints can't be built into the sampling; candidates
       drawn as above are kept only if they satisfy them (rejection sampling).

    Examples:
        >>> RandomStrategy(domain=domain, fallback_sampler=SobolSampler(), seed=42)
    """

    type: Literal["RandomStrategy"] = "RandomStrategy"
    fallback_sampler: AnySampler = Field(
        default=UniformSampler(),
        description="How inputs without constraints, and categorical and discrete "
        "inputs, are sampled.",
    )
    n_burnin: PositiveInt = Field(
        default=1000,
        description="Number of initial steps of the random walk that are discarded, "
        "so the samples no longer depend on its starting point.",
    )
    n_thinning: PositiveInt = Field(
        default=32,
        description="Only every n-th step of the random walk is kept, so consecutive "
        "samples are less correlated.",
    )
    num_base_samples: Optional[PositiveInt] = Field(
        default=None,
        description="Number of candidates drawn in each round of rejection sampling. "
        "If not provided, as many as are asked for.",
    )
    max_iters: PositiveInt = Field(
        default=1000,
        description="Maximal number of rounds of rejection sampling before the "
        "strategy gives up.",
    )
    max_combinations: PositiveInt = Field(
        default=64,
        description="Maximal number of distinct sets of active inputs drawn per ask "
        "under NChooseK constraints or `allow_zero` inputs. More sets mix the "
        "combinations better, at a higher cost.",
    )
    nchoosek_max_iters: PositiveInt = Field(
        default=1000,
        description="Maximal number of attempts to draw one valid set of active "
        "inputs under overlapping NChooseK constraints.",
    )

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
