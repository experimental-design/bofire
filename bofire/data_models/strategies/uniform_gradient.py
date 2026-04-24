from typing import Literal, Type

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
from bofire.data_models.strategies.strategy import Strategy


class UniformGradientStrategy(Strategy):
    """Data model for the uniform gradient program sampler.

    Samples monotone HPLC gradient programs using the following:
      - phi_0 ~ U[phi_min, phi_0_max]
      - phi_n ~ U[phi_last_min, phi_max]
      - interior phi knots: sorted uniforms in [phi_0, phi_n]
      - T (total duration) ~ U[t_last_min, t_max], or pinned to t_max if fixed
      - interior t knots: sorted uniforms in [t_min, T)

    All phi/t sampling parameters are derived directly from the domain's ContinuousInput
    feature bounds (created via ``create_absolute_domain(OptimizationSpec)``), so no
    duplicate fields are needed — except ``t_n_fixed``.

    # ``t_n_fixed``
    #     Required when the domain was built for fixed-duration mode (``t_last_min == t_max``
    #     in the spec): in that case ``t_{n_nodes}`` is not a free feature and the sampler
    #     needs to know what the fixed end time is.  Set it to the ``t_max`` value of the spec.
    #     Leave as ``None`` for variable-duration mode.
    """

    type: Literal["UniformGradientStrategy"] = "UniformGradientStrategy"

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
