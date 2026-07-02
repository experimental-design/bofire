from bofire.data_models.strategies.convergence_criteria._register import (  # noqa: F401
    register_convergence_criterion,
)
from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)
from bofire.data_models.strategies.convergence_criteria.objective_improvement import (
    ObjectiveImprovementCriterion,
)
from bofire.data_models.strategies.convergence_criteria.proposal_deviation import (
    ProposalDeviationCriterion,
)
from bofire.data_models.unions import tagged_union


_CONVERGENCE_CRITERION_TYPES: list[type[ConvergenceCriterion]] = [
    ObjectiveImprovementCriterion,
    ProposalDeviationCriterion,
]

AnyConvergenceCriterion = tagged_union(*_CONVERGENCE_CRITERION_TYPES)
