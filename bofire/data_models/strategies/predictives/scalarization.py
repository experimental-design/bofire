from typing import Any, Dict, List, Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.unions import tagged_union


class Scalarization(BaseModel):
    """How the objectives of several outputs are combined into the single value a
    single-objective strategy maximizes.

    The combination is applied to every sampled prediction of the outputs, so it can
    be any function of the objective values. Outputs with a constrained objective,
    such as a sigmoid one, either enter the combination like any objective or act as
    separate constraints the candidates should satisfy, depending on the
    scalarization.
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
        instead of entering the combination."""
        return False


def _validate_multiple_objectives(scalarization: Scalarization, domain: Domain):
    if len(domain.outputs.get_by_objective(Objective)) < 2:
        raise ValueError(
            f"{type(scalarization).__name__} requires at least 2 outputs with "
            "objectives. Consider SingleObjectiveScalarization instead.",
        )


class SingleObjectiveScalarization(Scalarization):
    """Optimizes the one output whose objective is not a constrained one.

    Further outputs with a constrained objective, such as a sigmoid one, act as
    constraints the candidates should satisfy.

    Examples:
        >>> SingleObjectiveScalarization()
    """

    type: Literal["SingleObjectiveScalarization"] = "SingleObjectiveScalarization"

    def separates_output_constraints(self) -> bool:
        return True


class _WeightedScalarization(Scalarization):
    """Combines the objectives of at least two outputs, each with a relative
    weight."""

    weights: Dict[str, PositiveFloat] = Field(
        default={},
        description="Relative importance of the objectives, by key of the output; "
        "outputs not listed have weight 1.",
    )

    def validate_domain(self, domain: Domain) -> None:
        _validate_multiple_objectives(self, domain)
        keys = domain.outputs.get_keys_by_objective(Objective)
        for key in self.weights:
            if key not in keys:
                raise ValueError(
                    f"Weight given for {key}, which is not an output with an "
                    "objective.",
                )

    def get_weights(self, domain: Domain) -> Dict[str, float]:
        """Returns the weight of every output with an objective.

        Args:
            domain: The domain of the strategy.

        Returns:
            The weights by output key, 1 for outputs not listed in `weights`.
        """
        return {
            key: self.weights.get(key, 1.0)
            for key in domain.outputs.get_keys_by_objective(Objective)
        }


class _MultiplicativeScalarization(_WeightedScalarization):
    """Combines the objectives by a product with the weights as exponents.

    The factors have to be non-negative: minimize and close-to-target objectives,
    whose values are negative, can't be multiplied, and a maximize objective, or a
    desirability without clipping, only stays non-negative within its bounds.
    """

    weight_normalization: Literal["min", "sum"] = Field(
        default="min",
        description="How the weights are rescaled before they become exponents: "
        "'min' so that the smallest is 1, 'sum' so that they sum to 1, which makes "
        "the product a weighted geometric mean. The best candidate is the same either "
        "way, but the acquisition values, and with them the proposals, differ.",
    )

    def _multiplied_keys(self, domain: Domain) -> List[str]:
        return domain.outputs.get_keys_by_objective(Objective)

    def validate_domain(self, domain: Domain) -> None:
        super().validate_domain(domain)
        for key in self._multiplied_keys(domain):
            objective = domain.outputs.get_by_key(key).objective
            if isinstance(objective, (MinimizeObjective, CloseToTargetObjective)):
                raise ValueError(
                    f"{type(self).__name__} can't multiply the objective of {key}: a "
                    f"{type(objective).__name__} has negative values. Use a "
                    "desirability or sigmoid objective instead, or WeightedSum"
                    "Scalarization.",
                )


class WeightedSumScalarization(_WeightedScalarization):
    """Maximizes the weighted sum of the objective values.

    A poor value of one objective can be made up for by good values of the others.

    Examples:
        >>> WeightedSumScalarization(weights={"yield": 2.0})
    """

    type: Literal["WeightedSumScalarization"] = "WeightedSumScalarization"
    use_output_constraints: bool = Field(
        default=True,
        description="Whether outputs with a constrained objective, such as a sigmoid "
        "one, act as constraints the candidates should satisfy. If not, they are "
        "summed like any objective.",
    )

    def separates_output_constraints(self) -> bool:
        return self.use_output_constraints


class WeightedProductScalarization(_MultiplicativeScalarization):
    """Maximizes the product of the objective values, each raised to the power of
    its weight.

    A candidate is only good if all objectives are good, since a value near zero in
    one of them makes the product small. Constrained objectives, such as sigmoid
    ones, are factors like any other.

    Examples:
        >>> WeightedProductScalarization(
        ...     weights={"yield": 2.0}, weight_normalization="sum"
        ... )
    """

    type: Literal["WeightedProductScalarization"] = "WeightedProductScalarization"


class WeightedProductSumScalarization(_MultiplicativeScalarization):
    """Maximizes a combination of objective values, partly by product and partly by
    sum.

    The objectives of the outputs in `additive_features` are summed, the others
    multiplied, each weighted by its rescaled weight. With multiplicative objectives
    f1 and f2 and additive objectives f3 and f4, the combined objective is

        f1^w1 * f2^w2 * (1 + w3 * f3 + w4 * f4)

    so the multiplicative objectives all have to be good, while the additive ones can
    make up for each other. Constrained objectives, such as sigmoid ones, enter the
    combination like any other.

    Examples:
        >>> WeightedProductSumScalarization(additive_features=["y3", "y4"])
    """

    type: Literal["WeightedProductSumScalarization"] = "WeightedProductSumScalarization"
    additive_features: List[str] = Field(
        default=[],
        description="Keys of the outputs whose objectives are summed; all other "
        "objectives are multiplied.",
    )

    def _multiplied_keys(self, domain: Domain) -> List[str]:
        return [
            key
            for key in super()._multiplied_keys(domain)
            if key not in self.additive_features
        ]

    def validate_domain(self, domain: Domain) -> None:
        output_keys = domain.outputs.get_keys()
        for key in self.additive_features:
            if key not in output_keys:
                raise ValueError(
                    f"Feature {key} is not an output feature of the domain."
                )
        super().validate_domain(domain)


_SCALARIZATION_TYPES: list[type[Scalarization]] = [
    SingleObjectiveScalarization,
    WeightedSumScalarization,
    WeightedProductScalarization,
    WeightedProductSumScalarization,
]

AnyScalarization = tagged_union(*_SCALARIZATION_TYPES)
