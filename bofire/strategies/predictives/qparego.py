from typing import Callable, List, Tuple, Union

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.strategies.api import QparegoStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import (
    get_multiobjective_objective,
    get_output_constraints,
    tkwargs,
)


# this implementation follows this tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/multi_objective_bo.ipynb
# main difference to the multiobjective strategies is that we have a randomized list of acqfs, this has to be bring into accordance
# with the other strategies
class QparegoStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function
        self.constraint_callables, self.etas = None, 1e-3

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        GenericMCObjective,
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        """Returns the scalarized objective.

        Returns:
            GenericMCObjective: the botorch objective.
            Union[ConstrainedObjective, None]: the botorch constraints.
            Union[List, float]: etas used in the botorch constraints.
        """
        ref_point_mask = torch.from_numpy(get_ref_point_mask(domain=self.domain)).to(
            **tkwargs
        )
        weights = (
            sample_simplex(
                len(
                    self.domain.outputs.get_keys_by_objective(
                        includes=[
                            MaximizeObjective,
                            MinimizeObjective,
                            CloseToTargetObjective,
                        ]
                    )
                ),
                **tkwargs,
            ).squeeze()
            * ref_point_mask
        )

        obj_callable = get_multiobjective_objective(outputs=self.domain.outputs)

        df_preds = self.predict(
            self.domain.outputs.preprocess_experiments_any_valid_output(
                experiments=self.experiments
            )
        )

        preds = torch.from_numpy(
            df_preds[[f"{key}_pred" for key in self.domain.outputs.get_keys()]].values
        ).to(**tkwargs)

        scalarization = get_chebyshev_scalarization(
            weights=weights, Y=obj_callable(preds, None) * ref_point_mask
        )

        def objective_callable(Z, X=None):
            return scalarization(obj_callable(Z, None) * ref_point_mask, X)

        if len(weights) != len(self.domain.outputs.get_by_objective(Objective)):
            constraint_callables, etas = get_output_constraints(self.domain.outputs)
        else:
            constraint_callables, etas = None, 1e-3

        return (
            GenericMCObjective(objective=objective_callable),
            constraint_callables,
            etas,
        )

    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        acqfs = []

        X_train, X_pending = self.get_acqf_input_tensors()

        (
            objective_callable,
            constraint_callables,
            etas,
        ) = self._get_objective_and_constraints()

        assert self.model is not None
        for i in range(n):
            acqf = get_acquisition_function(
                acquisition_function_name=self.acquisition_function.__class__.__name__,
                model=self.model,
                objective=objective_callable,
                X_observed=X_train,
                X_pending=X_pending if i == 0 else None,
                constraints=constraint_callables,
                eta=torch.tensor(etas).to(**tkwargs),
                mc_samples=self.num_sobol_samples,
                prune_baseline=True,
                cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            )
            acqfs.append(acqf)
        return acqfs
