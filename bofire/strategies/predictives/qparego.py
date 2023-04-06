from typing import Union

import pandas as pd
import torch
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.utils import get_acquisition_function
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.strategies.api import QparegoStrategy as DataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import (
    get_linear_constraints,
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

    def _init_acqf(self) -> None:
        pass

    def calc_acquisition(self, experiments: pd.DataFrame, combined: bool = False):
        raise ValueError("ACQF calc not implemented for qparego")

    def get_objective(
        self, pred: torch.Tensor
    ) -> Union[GenericMCObjective, ConstrainedMCObjective]:
        """Returns the scalarized objective.

        Args:
            pred (torch.Tensor): Predictions for the training data from the
                trained model.

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

        obj_callable = get_multiobjective_objective(output_features=self.domain.outputs)

        scalarization = get_chebyshev_scalarization(
            weights=weights, Y=obj_callable(pred, None) * ref_point_mask
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

    def _ask(self, candidate_count: int):
        assert candidate_count > 0, "candidate_count has to be larger than zero."

        # get the list acqfs
        acqf_list = []
        with torch.no_grad():
            clean_experiments = self.domain.outputs.preprocess_experiments_any_valid_output(
                self.experiments  # type: ignore
            )
            transformed = self.domain.inputs.transform(
                clean_experiments, self.input_preprocessing_specs
            )

            train_x = torch.from_numpy(transformed.values).to(**tkwargs)

            pred = self.model.posterior(train_x).mean  # type: ignore

        clean_experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments  # type: ignore
        )
        transformed = self.domain.inputs.transform(
            clean_experiments, self.input_preprocessing_specs
        )
        observed_x = torch.from_numpy(transformed.values).to(**tkwargs)

        # TODO: unite it with SOBO and also add the other acquisition functions
        for i in range(candidate_count):
            assert self.model is not None
            acqf = get_acquisition_function(
                acquisition_function_name="qNEI",
                model=self.model,
                objective=self.get_objective(pred),
                X_observed=observed_x,
                mc_samples=self.num_sobol_samples,
                qmc=True,
                prune_baseline=True,
            )
            acqf_list.append(acqf)

        # optimize
        (
            bounds,
            ic_generator,
            ic_gen_kwargs,
            nchooseks,
            fixed_features,
            fixed_features_list,
        ) = self._setup_ask()

        candidates, _ = optimize_acqf_list(
            acq_function_list=acqf_list,
            bounds=bounds,
            num_restarts=self.num_restarts,
            raw_samples=self.num_raw_samples,
            equality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
            ),
            inequality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
            ),
            nonlinear_inequality_constraints=nchooseks,
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
            ic_gen_kwargs=ic_gen_kwargs,
            ic_generator=ic_generator,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return self._postprocess_candidates(candidates=candidates)
