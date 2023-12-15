import base64
import warnings
from typing import Callable, List, Tuple, Union

try:
    import cloudpickle
except ModuleNotFoundError:
    warnings.warn(
        "Cloudpickle is not available. CustomSoboStrategy's `f` cannot be dumped or loaded."
    )

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.acquisition_functions.api import qLogNEI, qNEI, qPI, qSR, qUCB
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.api import AdditiveSoboStrategy as AdditiveDataModel
from bofire.data_models.strategies.api import CustomSoboStrategy as CustomDataModel
from bofire.data_models.strategies.api import (
    MultiplicativeSoboStrategy as MultiplicativeDataModel,
)
from bofire.data_models.strategies.predictives.sobo import SoboBaseStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import (
    get_additive_botorch_objective,
    get_custom_botorch_objective,
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

        (
            objective_callable,
            constraint_callables,
            etas,
        ) = self._get_objective_and_constraints()

        assert self.model is not None

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,
            objective_callable,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraint_callables,
            mc_samples=self.num_sobol_samples,
            beta=self.acquisition_function.beta
            if isinstance(self.acquisition_function, qUCB)
            else 0.2,
            tau=self.acquisition_function.tau
            if isinstance(self.acquisition_function, qPI)
            else 1e-3,
            eta=torch.tensor(etas).to(**tkwargs),
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            prune_baseline=self.acquisition_function.prune_baseline
            if isinstance(self.acquisition_function, (qNEI, qLogNEI))
            else True,
        )
        return [acqf]

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        try:
            target_feature = self.domain.outputs.get_by_objective(
                excludes=ConstrainedObjective
            )[0]
        except IndexError:
            target_feature = self.domain.outputs.get_by_objective(includes=Objective)[0]
        target_index = self.domain.outputs.get_keys().index(target_feature.key)
        objective_callable = get_objective_callable(
            idx=target_index, objective=target_feature.objective
        )

        # get the constraints
        if (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0) and (
            len(self.domain.outputs.get_by_objective(Objective)) > 1
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs
            )
        else:
            constraint_callables, etas = None, 1e-3

        # special cases of qUCB and qSR do not work with separate constraints
        if (isinstance(self.acquisition_function, (qSR, qUCB))) and (
            constraint_callables is not None
        ):
            return (
                ConstrainedMCObjective(
                    objective=objective_callable,  # type: ignore
                    constraints=constraint_callables,
                    eta=torch.tensor(etas).to(**tkwargs),
                    infeasible_cost=self.get_infeasible_cost(
                        objective=objective_callable
                    ),
                ),
                None,
                1e-3,
            )

        # return regular objective
        return (
            GenericMCObjective(objective=objective_callable),
            constraint_callables,
            etas,
        )


class AdditiveSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: AdditiveDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.use_output_constraints = data_model.use_output_constraints

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        # get the constraints
        if (
            (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0)
            and (len(self.domain.outputs.get_by_objective(Objective)) > 1)
            and self.use_output_constraints
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs
            )
        else:
            constraint_callables, etas = None, 1e-3
        # TODO: test this
        if self.use_output_constraints:
            objective_callable = get_additive_botorch_objective(
                outputs=self.domain.outputs, exclude_constraints=True
            )

            # special cases of qUCB and qSR do not work with separate constraints
            if isinstance(self.acquisition_function, (qSR, qUCB)):
                return (
                    ConstrainedMCObjective(
                        objective=objective_callable,  # type: ignore
                        constraints=constraint_callables,
                        eta=torch.tensor(etas).to(**tkwargs),
                        infeasible_cost=self.get_infeasible_cost(
                            objective=objective_callable
                        ),
                    ),
                    None,
                    1e-3,
                )
            else:
                return (
                    GenericMCObjective(objective=objective_callable),
                    constraint_callables,
                    etas,
                )

        # we absorb all constraints into the objective
        return (
            GenericMCObjective(
                objective=get_additive_botorch_objective(  # type: ignore
                    outputs=self.domain.outputs, exclude_constraints=False
                )
            ),
            constraint_callables,
            etas,
        )


class MultiplicativeSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: MultiplicativeDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        GenericMCObjective,
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        # we absorb all constraints into the objective
        return (
            GenericMCObjective(
                objective=get_multiplicative_botorch_objective(  # type: ignore
                    outputs=self.domain.outputs
                )
            ),
            None,
            1e-3,
        )


class CustomSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: CustomDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.use_output_constraints = data_model.use_output_constraints
        if data_model.dump is not None:
            self.loads(data_model.dump)
        else:
            self.f = None

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        if self.f is None:
            raise ValueError("No function has been provided for the strategy")
        # get the constraints
        if (
            (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0)
            and (len(self.domain.outputs.get_by_objective(Objective)) > 1)
            and self.use_output_constraints
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs
            )
        else:
            constraint_callables, etas = None, 1e-3

        if self.use_output_constraints:
            objective_callable = get_custom_botorch_objective(
                outputs=self.domain.outputs, f=self.f, exclude_constraints=True
            )
            # special cases of qUCB and qSR do not work with separate constraints
            if isinstance(self.acquisition_function, (qSR, qUCB)):
                return (
                    ConstrainedMCObjective(
                        objective=objective_callable,  # type: ignore
                        constraints=constraint_callables,
                        eta=torch.tensor(etas).to(**tkwargs),
                        infeasible_cost=self.get_infeasible_cost(
                            objective=objective_callable
                        ),
                    ),
                    None,
                    1e-3,
                )
            else:
                return (
                    GenericMCObjective(objective=objective_callable),
                    constraint_callables,
                    etas,
                )

        # we absorb all constraints into the objective
        return (
            GenericMCObjective(
                objective=get_custom_botorch_objective(
                    outputs=self.domain.outputs, f=self.f, exclude_constraints=False
                )
            ),
            constraint_callables,
            etas,
        )

    def dumps(self) -> str:
        """Dumps the function to a string via pickle as this is not directly json serializable."""
        if self.f is None:
            raise ValueError("No function has been provided for the strategy")
        f_bytes_dump = cloudpickle.dumps(self.f)
        return base64.b64encode(f_bytes_dump).decode()

    def loads(self, data: str):
        """Loads the function from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        f_bytes_load = base64.b64decode(data.encode())
        self.f = cloudpickle.loads(f_bytes_load)
