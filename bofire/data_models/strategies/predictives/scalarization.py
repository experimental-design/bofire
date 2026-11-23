from typing import Any, Dict, List, Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    ConstrainedObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.unions import tagged_union


class Scalarization(BaseModel):
    """How the objectives of several outputs are combined into the single value a
    single-objective strategy maximizes.

    The scalarization is applied to every sampled prediction of the outputs, so it
    can be any function of the objective values.
    """

    type: Any

    def validate_domain(self, domain: Domain) -> None:
        """Checks that the scalarization can combine the objectives of the domain.

        Args:
            domain: The domain of the strategy.

        Raises:
            ValueError: if the domain's objectives don't suit the scalarization.
        """

    def separates_output_constraints(self) -> bool:
        """Whether outputs with a constrained objective act as separate constraints
        instead of entering the scalarization."""
        return True


class WeightedScalarization(Scalarization):
    """Combines several objectives by a weighted product and a weighted sum.

    The objectives of the outputs in `additive_features` are summed, the others
    multiplied, each weighted by its weight; the weights are normalized to sum to 1.
    With multiplied objectives f1 and f2 and summed objectives f3 and f4, the
    scalarization is

        f1^w1 * f2^w2 * (1 + w3 * f3 + w4 * f4)

    so a candidate is only good if every multiplied objective is good, while the
    summed ones can make up for each other. Without `additive_features` this is a
    weighted geometric mean; with all objectives in `additive_features`, a weighted
    sum. Multiplied objectives have to be non-negative: minimize and
    close-to-target objectives can only be summed, and a maximize objective, or a
    desirability without clipping, is only non-negative within its bounds.

    Examples:
        >>> WeightedScalarization(weights={"yield": 2.0})
        >>> WeightedScalarization(additive_features=["yield", "cost"])
    """

    type: Literal["WeightedScalarization"] = "WeightedScalarization"
    weights: Dict[str, PositiveFloat] = Field(
        default={},
        description="Relative importance of the objectives, by key of the output; "
        "outputs not listed have weight 1.",
    )
    additive_features: List[str] = Field(
        default=[],
        description="Keys of the outputs whose objectives are summed; all other "
        "objectives are multiplied.",
    )
    use_output_constraints: bool = Field(
        default=True,
        description="Whether outputs with a constrained objective, such as a sigmoid "
        "one, act as constraints the candidates should satisfy. If not, they enter "
        "the scalarization like any objective, e.g. as desirabilities that are "
        "multiplied.",
    )

    def separates_output_constraints(self) -> bool:
        return self.use_output_constraints

    def _scalarized_keys(self, domain: Domain) -> List[str]:
        if self.use_output_constraints:
            return domain.outputs.get_keys_by_objective(excludes=ConstrainedObjective)
        return domain.outputs.get_keys_by_objective(Objective)

    def validate_domain(self, domain: Domain) -> None:
        keys = self._scalarized_keys(domain)
        if len(keys) < 2:
            raise ValueError(
                "WeightedScalarization requires at least 2 objectives to combine. "
                "For a single objective, leave the scalarization unset.",
            )
        for name, given in [
            ("Weight", self.weights),
            ("Additive feature", self.additive_features),
        ]:
            for key in given:
                if key not in keys:
                    raise ValueError(
                        f"{name} given for {key}, which is not an objective of the "
                        "scalarization.",
                    )
        for key in keys:
            if key in self.additive_features:
                continue
            objective = domain.outputs.get_by_key(key).objective
            if isinstance(objective, (MinimizeObjective, CloseToTargetObjective)):
                raise ValueError(
                    f"WeightedScalarization can't multiply the objective of {key}: a "
                    f"{type(objective).__name__} has negative values. Add it to "
                    "`additive_features`, or use a desirability or sigmoid objective.",
                )

    def get_weights(self, domain: Domain) -> Dict[str, float]:
        """Returns the weight of every objective of the scalarization.

        Args:
            domain: The domain of the strategy.

        Returns:
            The weights by output key, 1 for outputs not listed in `weights`.
        """
        return {
            key: self.weights.get(key, 1.0) for key in self._scalarized_keys(domain)
        }


_SCALARIZATION_TYPES: list[type[Scalarization]] = [WeightedScalarization]

AnyScalarization = tagged_union(*_SCALARIZATION_TYPES)
