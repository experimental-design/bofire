from typing import Callable, List, Tuple, Union, cast

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
)
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    AnyUnconstrainedAcquisitionFunction,
    qLogPF,
)
from bofire.data_models.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.api import AnyScalarization
from bofire.data_models.strategies.convergence_criteria.api import (
    AnyConvergenceCriterion,
)
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.strategies.predictives.sobo import SoboStrategy as SoboDataModel
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.data_models.unions import unwrap_annotated
from bofire.strategies.predictives import scalarization
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.strategy import make_strategy
from bofire.utils.torch_tools import get_output_constraints, tkwargs


class SoboStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: SoboDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self.scalarization = data_model.scalarization

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        X_train, X_pending = self.get_acqf_input_tensors()

        (
            objective_callable,
            constraint_callables,
            etas,
        ) = self._get_objective_and_constraints()

        assert self.model is not None

        params = self.acquisition_function.model_dump()

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,
            objective_callable,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraint_callables,
            mc_samples=params.get("n_mc_samples", 512),
            beta=params.get("beta", 0.2),
            tau=params.get("tau", 1e-3),
            eta=torch.tensor(etas).to(**tkwargs),
            cache_root=None,
            prune_baseline=params.get("prune_baseline", True),
        )
        return [acqf]

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective, IdentityMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        assert self.experiments is not None, "No experiments available."
        objective_callable = scalarization.map(
            self.scalarization,
            domain=self.domain,
            experiments=self.experiments,
            input_preprocessing_specs=self.input_preprocessing_specs,
        )

        # constrained objectives are separate constraints only if the scalarization
        # keeps them apart and there is something else to optimize
        if (
            self.scalarization.separates_output_constraints()
            and len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0
            and len(self.domain.outputs.get_by_objective(Objective)) > 1
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs,
                experiments=self.experiments,
            )
        else:
            constraint_callables, etas = None, 1e-3

        # special cases of qUCB and qSR do not work with separate constraints
        if isinstance(
            self.acquisition_function,
            unwrap_annotated(AnyUnconstrainedAcquisitionFunction)[0],
        ) and (constraint_callables is not None):
            return (
                ConstrainedMCObjective(
                    objective=objective_callable,
                    constraints=constraint_callables,
                    eta=torch.tensor(etas).to(**tkwargs),
                    infeasible_cost=self.get_infeasible_cost(
                        objective=objective_callable,
                    ),
                ),
                None,
                1e-3,
            )

        return (
            GenericMCObjective(objective=objective_callable)
            if not isinstance(self.acquisition_function, qLogPF)
            else IdentityMCObjective(),
            constraint_callables,
            etas,
        )

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: AnySingleObjectiveAcquisitionFunction
        | qLogPF
        | None = None,
        scalarization: AnyScalarization | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        seed: int | None = None,
        include_infeasible_exps_in_acqf_calc: bool | None = False,
        convergence_criterion: AnyConvergenceCriterion | None = None,
    ) -> Self:
        """
        Creates a single objective Bayesian optimization strategy.
        Args:
            domain: The optimization domain of the strategy.
            acquisition_function: The acquisition function to use.
            scalarization: How the objectives are combined into one; by default the
                domain has a single objective to optimize.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            seed: The random seed to use.
            include_infeasible_exps_in_acqf_calc: Whether infeasible experiments should be included in the set
                of experiments used to compute the acquisition function.
        """
        return cast(Self, make_strategy(cls, SoboDataModel, locals()))
