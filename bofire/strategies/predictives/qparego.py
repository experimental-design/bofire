from typing import Callable, List, Tuple, Union, cast

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from typing_extensions import Self

from bofire.data_models.acquisition_functions.acquisition_function import (
    qEI,
    qLogEI,
    qLogNEI,
    qNEI,
)
from bofire.data_models.api import Domain
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.api import QparegoStrategy as DataModel
from bofire.data_models.strategies.convergence_criteria.api import (
    AnyConvergenceCriterion,
)
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.strategy import make_strategy
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import (
    get_multiobjective_objective,
    get_output_constraints,
    tkwargs,
)


class QparegoStrategy(BotorchStrategy):
    """Generate candidates sequentially with an independent scalarization per point."""

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self.constraint_callables, self.etas = None, 1e-3

    def _get_objectives_and_constraints(
        self,
        n: int,
    ) -> Tuple[
        List[GenericMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        """Draw independent scalarizations using shared baseline predictions.

        Each candidate gets its own weights and callable. Output transformations,
        posterior-mean predictions, and output constraints are computed once.
        """
        assert self.experiments is not None, "No experiments available."
        ref_point_mask = torch.from_numpy(get_ref_point_mask(domain=self.domain)).to(
            **tkwargs,
        )
        obj_callable = get_multiobjective_objective(
            outputs=self.domain.outputs,
            experiments=self.experiments,
        )

        df_preds = self.predict(
            self.domain.outputs.preprocess_experiments_any_valid_output(
                experiments=self.experiments,
            ),
        )

        preds = torch.from_numpy(
            df_preds[[f"{key}_pred" for key in self.domain.outputs.get_keys()]].values,
        ).to(**tkwargs)

        Y = obj_callable(preds, None) * ref_point_mask

        def make_objective() -> GenericMCObjective:
            weights = (
                sample_simplex(len(ref_point_mask), **tkwargs).squeeze()
                * ref_point_mask
            )
            scalarization = get_chebyshev_scalarization(weights=weights, Y=Y)

            def objective_callable(Z, X=None):
                return scalarization(obj_callable(Z, None) * ref_point_mask, X)

            return GenericMCObjective(objective=objective_callable)

        if len(ref_point_mask) != len(self.domain.outputs.get_by_objective(Objective)):
            constraint_callables, etas = get_output_constraints(
                self.domain.outputs,
                experiments=self.experiments,
            )
        else:
            constraint_callables, etas = None, 1e-3

        return (
            [make_objective() for _ in range(n)],
            constraint_callables,
            etas,
        )

    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        acqfs = []

        X_train, X_pending = self.get_acqf_input_tensors()

        (
            objectives,
            constraint_callables,
            etas,
        ) = self._get_objectives_and_constraints(n)

        assert self.model is not None
        for i, objective in enumerate(objectives):
            acqf = get_acquisition_function(
                acquisition_function_name=self.acquisition_function.__class__.__name__,
                model=self.model,
                objective=objective,
                X_observed=X_train,
                X_pending=X_pending if i == 0 else None,
                constraints=constraint_callables,
                eta=torch.tensor(etas).to(**tkwargs),
                mc_samples=self.acquisition_function.n_mc_samples,
                prune_baseline=True,
                cache_root=None,
            )
            acqfs.append(acqf)
        return acqfs

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: qEI | qLogEI | qLogNEI | qNEI | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        seed: int | None = None,
        include_infeasible_exps_in_acqf_calc: bool | None = False,
        convergence_criterion: AnyConvergenceCriterion | None = None,
    ) -> Self:
        """
        Creates an instance of the multi-objective strategy ParEGO using the provided configuration parameters.

        J. Knowles. Parego: a hybrid algorithm with on-line landscape approximation for expensive multiobjective
        optimization problems. IEEE Transactions on Evolutionary Computation, 10(1):50-66, 2006

        S. Daulton, M. Balandat, and E. Bakshy.
        Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization.
        Advances in Neural Information Processing Systems 33, 2020.

        Args:
            domain: The optimization domain of the strategy.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer for the acquisition function.
            surrogate_specs: Specifications for the surrogate model.
            seed: Random seed for reproducibility.
            include_infeasible_exps_in_acqf_calc: Whether infeasible experiments should be included in the set
                of experiments used to compute the acquisition function.
        Returns:
            An instance of the strategy configured with the provided parameters.
        """
        return cast(Self, make_strategy(cls, DataModel, locals()))
