import warnings
from abc import abstractmethod
from typing import Literal, Optional, Type

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints import api as constraints
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import BotorchSurrogates
from bofire.data_models.types import IntPowerOfTwo
from bofire.data_models.unions import tagged_union


class AcquisitionOptimizer(BaseModel):
    """Method that maximizes the acquisition function over the domain, i.e. finds
    the candidates to propose."""

    prefer_exhaustive_search_for_purely_categorical_domains: bool = Field(
        default=True,
        description="Whether, if every input is categorical or discrete, all "
        "combinations are scored instead of running the optimizer. Required for "
        "categorical exclude constraints.",
    )

    @abstractmethod
    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented.

        Args:
            my_type (Type[Feature]): The type of the constraint.

        Returns:
            bool: True if the constraint is implemented, False otherwise.

        """
        pass

    @abstractmethod
    def validate_domain(self, domain: Domain):
        """Validates the fit of the domain to the optimizer.

        Args:
            domain (Domain): The domain to be validated.
        """
        pass

    @abstractmethod
    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        """Validates the surrogate specs for the optimizer.

        Args:
            surrogate_specs: The surrogate specs to be validated.

        Returns:
            SurrogateSpecs: The validated surrogate specs.

        """
        pass


class LocalSearchConfig(BaseModel):
    """LocalSearchConfigs provide a way to define how to switch between global
    acqf optimization in the global bounds and local acqf optimization in the local
    reference bounds.
    """

    @abstractmethod
    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        """Abstract switching function between local and global acqf optimum.

        Args:
            acqf_local (float): Local acqf value.
            acqf_global (float): Global acqf value.

        Returns:
            bool: If true, do local step, else a step towards the global acqf maximum.

        """


class LSRBO(LocalSearchConfig):
    """Local search region Bayesian optimization (LSR-BO,
    https://www.merl.com/publications/docs/TR2023-057.pdf).

    For processes that may only move by a limited step from the current point, as
    set by `local_relative_bounds` on the continuous inputs. Each step maximizes the
    acquisition function within the local region; if the value found there is too
    low, the step instead heads for the global maximum along a feasible path. Only
    used when one candidate is asked for.

    Examples:
        >>> BotorchOptimizer(local_search_config=LSRBO(gamma=0.1))
    """

    type: Literal["LSRBO"] = "LSRBO"
    gamma: float = Field(
        default=0.1,
        description="Threshold on the acquisition value at the local maximum: if it "
        "is at least `gamma`, the local step is taken, otherwise a step towards the "
        "global maximum. The default suits qEI; it has to be adapted to the "
        "acquisition function, especially to log-based ones.",
    )

    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        return acqf_local >= self.gamma


AnyLocalSearchConfig = LSRBO


class BotorchOptimizer(AcquisitionOptimizer):
    """Gradient-based optimization of the acquisition function from several starting
    points, with BoTorch.

    Continuous inputs are optimized by gradients. Categorical and discrete inputs are
    handled by trying every combination of their values if there are few, and
    otherwise by optimizing them alternately with the continuous inputs. Supports all
    constraints except nonlinear ones.

    Examples:
        >>> BotorchOptimizer(n_restarts=10, n_raw_samples=512)
    """

    type: Literal["BotorchOptimizer"] = "BotorchOptimizer"
    n_restarts: PositiveInt = Field(
        default=20,
        description="Number of starting points from which the optimization is run. "
        "More starts find better maxima at a higher cost.",
    )
    n_raw_samples: IntPowerOfTwo = Field(
        default=1024,
        description="Number of random points at which the acquisition function is "
        "evaluated to choose the starting points; a power of two.",
    )
    maxiter: PositiveInt = Field(
        default=2000,
        description="Maximal number of iterations of each optimization run.",
    )
    batch_limit: Optional[PositiveInt] = Field(
        default=None,
        validate_default=True,
        description="Number of starts optimized together in one batch; smaller "
        "batches need less memory. At most, and if not provided equal to, "
        "`n_restarts`.",
    )
    sequential: bool = Field(
        default=False,
        description="Whether the candidates of a batch are chosen one after another, "
        "each given the ones before, instead of all jointly. Cheaper for large "
        "batches and often as good, see "
        "https://github.com/pytorch/botorch/discussions/2810.",
    )
    local_search_config: Optional[AnyLocalSearchConfig] = Field(
        default=None,
        description="Rule for taking steps within a local search region. Requires "
        "a domain with only linear constraints and without semi-continuous inputs; "
        "if not provided, every step maximizes over the whole domain.",
    )

    # NChooseK / semi-continuous pruning hyperparameters. See
    # `bofire.strategies.predictives._nchoosek_pruning.prune_nchoosek` for
    # full semantics.
    per_step_local_reopt: bool = Field(
        default=False,
        description=(
            "When pruning is applicable, refine each per-feature variant "
            "via optimize_acqf in addition to the QP projection. More "
            "accurate but multiplies cost by the local solver."
        ),
    )
    final_local_reopt: bool = Field(
        default=True,
        description=(
            "When pruning is applicable, run a single optimize_acqf "
            "clean-up after the greedy loop with zeroed and active "
            "semi-continuous features pinned."
        ),
    )

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["n_restarts"],
            info.data["n_restarts"],
        )
        return batch_limit

    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        pass

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented. Currently only linear constraints are supported.

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        if my_type in [
            constraints.NonlinearInequalityConstraint,
            constraints.NonlinearEqualityConstraint,
        ]:
            return False
        return True

    def validate_domain(self, domain: Domain):
        def validate_local_search_config(domain: Domain):
            if self.local_search_config is not None:
                if has_local_search_region(domain) is False:
                    warnings.warn(
                        "`local_search_region` config is specified, but no local search region is defined in `domain`",
                    )
                if (
                    len(domain.constraints)
                    - len(domain.constraints.get(constraints.LinearConstraint))
                    > 0
                ):
                    raise ValueError("LSR-BO only supported for linear constraints.")
                # Semi-continuous features (`allow_zero=True` with
                # `bounds[0] > 0`) create a disconnected feasible region
                # `{0} ∪ [lb, ub]` that LSR-BO's QP-based shortest-path
                # cannot interpolate across without producing infeasible
                # intermediate steps. The local AF solver would also
                # produce fractional candidates in the gap `(0, lb)`.
                for feat in domain.inputs.get(ContinuousInput):
                    assert isinstance(feat, ContinuousInput)
                    if feat.is_semicontinuous:
                        raise ValueError(
                            "LSR-BO is not supported for domains with "
                            "semi-continuous features (`allow_zero=True` "
                            "with `bounds[0] > 0`). Feature "
                            f"{feat.key!r} violates this."
                        )

        def validate_exclude_constraints(domain: Domain):
            if (
                len(domain.constraints.get(constraints.CategoricalExcludeConstraint))
                > 0
            ):
                if len(
                    domain.inputs.get([CategoricalInput, DiscreteInput]),
                ) != len(domain.inputs):
                    raise ValueError(
                        "CategoricalExcludeConstraints can only be used for pure categorical/discrete search spaces.",
                    )
                if (
                    self.prefer_exhaustive_search_for_purely_categorical_domains
                    is False
                ):
                    raise ValueError(
                        "CategoricalExcludeConstraints can only be used with exhaustive search for purely categorical/discrete search spaces.",
                    )

        validate_local_search_config(domain)
        validate_exclude_constraints(domain)


class GeneticAlgorithmOptimizer(AcquisitionOptimizer):
    """Genetic algorithm for the acquisition function, with pymoo's mixed-variable
    algorithm.

    A population of candidates evolves over generations by mutation, crossover and
    selection. It needs no gradients, so it handles categorical and discrete inputs
    directly and supports nonlinear inequality constraints, which the gradient-based
    optimizer cannot, at a higher cost. Only for
    single-objective acquisition functions. The run stops at the first termination
    criterion met.

    Examples:
        >>> GeneticAlgorithmOptimizer(population_size=500, n_max_gen=200)
    """

    type: Literal["GeneticAlgorithm"] = "GeneticAlgorithm"

    population_size: PositiveInt = Field(
        default=1000,
        description="Number of candidates in each generation.",
    )
    xtol: PositiveFloat = Field(
        default=0.0005,
        description="The run stops once the inputs of the population change less "
        "than this between generations.",
    )
    cvtol: PositiveFloat = Field(
        default=1e-8,
        description="The run stops once the constraint violation changes less than "
        "this between generations.",
    )
    ftol: PositiveFloat = Field(
        default=1e-6,
        description="The run stops once the acquisition values change less than this "
        "between generations.",
    )
    n_max_gen: PositiveInt = Field(
        default=500,
        description="Maximal number of generations.",
    )
    n_max_evals: PositiveInt = Field(
        default=100000,
        description="Maximal number of evaluations of the acquisition function.",
    )
    verbose: bool = Field(
        default=False,
        description="Whether the progress of the generations is printed.",
    )
    ga_progress_csv_path: Optional[str] = Field(
        default=None,
        description="Path to a CSV file where the progress of the genetic algorithm will be saved.",
    )

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        return my_type in [
            constraints.LinearEqualityConstraint,
            constraints.LinearInequalityConstraint,
            constraints.ProductInequalityConstraint,
            constraints.NonlinearInequalityConstraint,
            constraints.NChooseKConstraint,
            constraints.CategoricalExcludeConstraint,
        ]

    def validate_domain(self, domain: Domain):
        pass

    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        pass


AnyAcqfOptimizer = tagged_union(BotorchOptimizer, GeneticAlgorithmOptimizer)
