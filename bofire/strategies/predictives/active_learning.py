from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, get_acquisition_function
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
import torch
from botorch.acquisition import get_acquisition_function, qNegIntegratedPosteriorVariance
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.get_sampler import get_sampler

from bofire.data_models.acquisition_functions.api import qNegIntPosVar
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.predictives.sobo import SoboBaseStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import (
    get_objective_callable,
    get_output_constraints,
    tkwargs,
)


class ActiveLearningStrategy(BotorchStrategy):
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

        # sample mc points for integration
        sampler = get_sampler(
            posterior=self.model.posterior(X_train[:1]),
            sample_shape=torch.Size([self.num_sobol_samples]),
            seed=None,
        )
        mc_samples = sampler.forward(posterior=self.model.posterior(X_train[:1]))

        acqf = qNegIntegratedPosteriorVariance(
            model=self.model,
            mc_points=mc_samples,
            sampler=sampler,
            X_pending=X_pending,
        )

        # TODO: Use get_acquisition_function method as soon as qNegPostVar is implemented there by botorch.
        # acqf = get_acquisition_function(
        #     acquisition_function_name=self.acquisition_function.__class__.__name__,
        #     model=self.model,
        #     objective=objective_callable,  # TODO: include posterior transform for active learning mobo
        #     X_observed=X_train,
        #     X_pending=X_pending,
        #     # constraints=constraint_callables,
        #     # eta=torch.tensor(etas).to(**tkwargs),
        #     mc_samples=self.num_sobol_samples,
        #     cache_root=True if isinstance(self.model, GPyTorchModel) else False,
        # )
        return [acqf]

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        """Analyzes the domain's objectives and constraints to determine the appropriate objective function
        and constraints for optimization.

        Returns:
            Tuple[Union[GenericMCObjective, ConstrainedMCObjective],
                Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
                Union[List, float]]:
            A tuple containing:
            - The objective function, either a GenericMCObjective or ConstrainedMCObjective.
            - The constraint callables if any, otherwise None.
            - Eta values for constraints, a float or a list of floats.

        Raises:
            IndexError: If no suitable target feature is found in the domain's outputs.


        Returns:
            Tuple[ Union[GenericMCObjective, ConstrainedMCObjective], Union[List[Callable[[torch.Tensor], torch.Tensor]], None], Union[List, float], ]: _description_
        """
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

        # return regular objective
        return (
            GenericMCObjective(objective=objective_callable),
            constraint_callables,
            etas,
        )
