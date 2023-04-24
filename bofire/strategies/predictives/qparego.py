from typing import List, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.utils import get_acquisition_function
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
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

    def _get_objective(
        self,
    ) -> Union[GenericMCObjective, ConstrainedMCObjective]:
        """Returns the scalarized objective.

        Returns:
            Union[GenericMCObjective, ConstrainedMCObjective]: the botorch objective.
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

        def objective(Z, X=None):
            return scalarization(obj_callable(Z, None) * ref_point_mask, X)

        if len(weights) != len(self.domain.outputs):
            constraints, etas = get_output_constraints(self.domain.outputs)
            return ConstrainedMCObjective(
                objective=objective,
                constraints=constraints,
                eta=torch.tensor(etas).to(**tkwargs),
                infeasible_cost=self.get_infeasible_cost(objective=objective),
            )
        return GenericMCObjective(scalarization)

    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        acqfs = []

        X_train, X_pending = self.get_acqf_input_tensors()

        assert self.model is not None
        for i in range(n):
            acqf = get_acquisition_function(
                acquisition_function_name="qNEI",
                model=self.model,
                objective=self._get_objective(),
                X_observed=X_train,
                X_pending=X_pending if i == 0 else None,
                mc_samples=self.num_sobol_samples,
                qmc=True,
                prune_baseline=True,
            )
            acqfs.append(acqf)
        return acqfs
