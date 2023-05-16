from typing import List, Union

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.acquisition_functions.api import qPI, qUCB
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.strategies.api import AdditiveSoboStrategy as AdditiveDataModel
from bofire.data_models.strategies.api import (
    MultiplicativeSoboStrategy as MultiplicativeDataModel,
)
from bofire.data_models.strategies.predictives.sobo import SoboBaseStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import (
    get_additive_botorch_objective,
    get_multiplicative_botorch_objective,
    get_objective_callable,
    get_output_constraints,
    tkwargs,
)


class SoboStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        X_train, X_pending = self.get_acqf_input_tensors()

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,  # type: ignore
            self._get_objective(),
            X_observed=X_train,
            X_pending=X_pending,
            constraints=None,
            mc_samples=self.num_sobol_samples,
            qmc=True,
            beta=self.acquisition_function.beta
            if isinstance(self.acquisition_function, qUCB)
            else 0.2,
            tau=self.acquisition_function.tau
            if isinstance(self.acquisition_function, qPI)
            else 1e-3,
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
        )
        return [acqf]

    def _get_objective(self) -> GenericMCObjective:
        # TODO: test this
        # here we get the actual objective
        target_feature = self.domain.outputs.get_by_objective(
            excludes=ConstrainedObjective
        )[0]
        target_index = self.domain.outputs.get_keys().index(target_feature.key)
        objective_callable = get_objective_callable(
            idx=target_index, objective=target_feature.objective
        )

        # in case that constraints are present we return a constrained botorch objective
        if len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0:
            constraints, etas = get_output_constraints(outputs=self.domain.outputs)

            return ConstrainedMCObjective(
                objective=objective_callable,
                constraints=constraints,
                eta=torch.tensor(etas).to(**tkwargs),
                infeasible_cost=self.get_infeasible_cost(objective=objective_callable),
            )

        # else the generic one
        return GenericMCObjective(objective=objective_callable)


class AdditiveSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: AdditiveDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.use_output_constraints = data_model.use_output_constraints

    def _get_objective(self) -> Union[GenericMCObjective, ConstrainedMCObjective]:
        # TODO: test this
        if (
            self.use_output_constraints
            and len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0
        ):
            constraints, etas = get_output_constraints(outputs=self.domain.outputs)
            objective = get_additive_botorch_objective(
                outputs=self.domain.outputs, exclude_constraints=True
            )
            return ConstrainedMCObjective(
                objective=objective,  # type: ignore
                constraints=constraints,
                eta=torch.tensor(etas).to(**tkwargs),
                infeasible_cost=self.get_infeasible_cost(objective=objective),
            )
        return GenericMCObjective(
            objective=get_additive_botorch_objective(  # type: ignore
                outputs=self.domain.outputs, exclude_constraints=False
            )
        )


class MultiplicativeSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: MultiplicativeDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _get_objective(self) -> GenericMCObjective:
        return GenericMCObjective(
            objective=get_multiplicative_botorch_objective(  # type: ignore
                outputs=self.domain.outputs
            )
        )
