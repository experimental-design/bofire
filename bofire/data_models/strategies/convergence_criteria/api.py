from bofire.data_models.strategies.convergence_criteria._register import (  # noqa: F401
    register_convergence_criterion,
)
from bofire.data_models.strategies.convergence_criteria.convergence_criterion import (
    ConvergenceCriterion,
)
from bofire.data_models.strategies.convergence_criteria.exp_min_regret_gap import (
    ExpMinRegretGapCriterion,
)
from bofire.data_models.strategies.convergence_criteria.hypervolume_improvement import (
    HypervolumeImprovementCriterion,
)
from bofire.data_models.strategies.convergence_criteria.log_eipc import LogEipcCriterion
from bofire.data_models.strategies.convergence_criteria.objective_improvement import (
    ObjectiveImprovementCriterion,
)
from bofire.data_models.strategies.convergence_criteria.probabilistic_regret_bound import (
    ProbabilisticRegretBoundCriterion,
)
from bofire.data_models.strategies.convergence_criteria.proposal_deviation import (
    ProposalDeviationCriterion,
)
from bofire.data_models.strategies.convergence_criteria.ucb_lcb_regret_bound import (
    UcbLcbRegretBoundCriterion,
)
from bofire.data_models.unions import tagged_union


_CONVERGENCE_CRITERION_TYPES: list[type[ConvergenceCriterion]] = [
    ObjectiveImprovementCriterion,
    HypervolumeImprovementCriterion,
    ProposalDeviationCriterion,
    UcbLcbRegretBoundCriterion,
    ExpMinRegretGapCriterion,
    LogEipcCriterion,
    ProbabilisticRegretBoundCriterion,
]

AnyConvergenceCriterion = tagged_union(*_CONVERGENCE_CRITERION_TYPES)
