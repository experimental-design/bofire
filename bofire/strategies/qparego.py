from typing import Union

import numpy as np
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
    NChooseKConstraint,
)
from bofire.data_models.enum import CategoricalMethodEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import QparegoStrategy as DataModel
from bofire.strategies.botorch import BotorchStrategy
from bofire.strategies.multiobjective import get_ref_point_mask
from bofire.surrogates.torch_tools import (
    get_linear_constraints,
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

    def _init_domain(self) -> None:
        # first part of this is doubled with qehvi --> maybe create a common base class
        # this has to go into the validators
        if (
            len(
                self.domain.outputs.get_by_objective(
                    includes=[MaximizeObjective, MinimizeObjective]
                )
            )
            < 2
        ):
            raise ValueError(
                "At least two features with objective type `MaximizeObjective` or `MinimizeObjective` has to be defined in the domain."
            )
        for feat in self.domain.outputs.get_by_objective(excludes=None):
            if feat.objective.w != 1.0:  # type: ignore
                raise ValueError(
                    "Only objective functions with weight 1 are supported."
                )

        super()._init_domain()
        return

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
                        includes=[MaximizeObjective, MinimizeObjective]
                    )
                ),
                **tkwargs,
            ).squeeze()
            * ref_point_mask
        )
        key2indices = {key: i for i, key in enumerate(self.domain.outputs.get_keys())}
        indices = torch.tensor(
            [
                key2indices[key]
                for key in self.domain.outputs.get_keys_by_objective(
                    includes=[MaximizeObjective, MinimizeObjective]
                )
            ],
            dtype=torch.int64,
        )

        scalarization = get_chebyshev_scalarization(
            weights=weights, Y=pred[..., indices]
        )

        def objective(Z, X=None):
            return scalarization(Z[..., indices], X)

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
        lower, upper = self.domain.inputs.get_bounds(self.input_preprocessing_specs)

        num_categorical_features = len(self.domain.get_features(CategoricalInput))
        num_categorical_combinations = len(
            self.domain.inputs.get_categorical_combinations()
        )

        fixed_features = None
        fixed_features_list = None

        if (
            (num_categorical_features == 0)
            or (num_categorical_combinations == 1)
            or (
                (self.categorical_method == CategoricalMethodEnum.FREE)
                and (self.descriptor_method == CategoricalMethodEnum.FREE)
            )
        ) and len(self.domain.cnstrs.get(NChooseKConstraint)) == 0:
            fixed_features = self.get_fixed_features()

        elif (
            (self.categorical_method == CategoricalMethodEnum.EXHAUSTIVE)
            or (self.descriptor_method == CategoricalMethodEnum.EXHAUSTIVE)
        ) and len(self.domain.cnstrs.get(NChooseKConstraint)) == 0:
            fixed_features_list = self.get_categorical_combinations()

        elif len(self.domain.cnstrs.get(NChooseKConstraint)) > 0:
            fixed_features_list = self.get_fixed_values_list()
        else:
            raise IOError()

        candidates, _ = optimize_acqf_list(
            acq_function_list=acqf_list,
            bounds=torch.tensor([lower, upper]).to(**tkwargs),
            num_restarts=self.num_restarts,
            raw_samples=self.num_raw_samples,
            equality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
            ),
            inequality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
            ),
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
            options={"batch_limit": 5, "maxiter": 200},
        )

        preds = self.model.posterior(X=candidates).mean.detach().numpy()  # type: ignore
        stds = np.sqrt(self.model.posterior(X=candidates).variance.detach().numpy())  # type: ignore

        input_feature_keys = [
            item
            for key in self.domain.inputs.get_keys()
            for item in self._features2names[key]
        ]

        df_candidates = pd.DataFrame(
            data=candidates.detach().numpy(),
            columns=input_feature_keys,
        )

        df_candidates = self.domain.inputs.inverse_transform(
            df_candidates, self.input_preprocessing_specs
        )

        for i, feat in enumerate(self.domain.outputs.get_by_objective(excludes=None)):
            df_candidates[feat.key + "_pred"] = preds[:, i]
            df_candidates[feat.key + "_sd"] = stds[:, i]
            df_candidates[feat.key + "_des"] = feat.objective(preds[:, i])  # type: ignore

        return df_candidates
