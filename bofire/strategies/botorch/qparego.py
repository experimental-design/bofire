from typing import Type

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.utils import get_acquisition_function
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from bofire.domain.constraints import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.features import CategoricalDescriptorInput, CategoricalInput, Feature
from bofire.domain.objectives import (
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.strategies.botorch import tkwargs
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.utils.enum import AcquisitionFunctionEnum, CategoricalMethodEnum
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import get_linear_constraints


# this implementation follows this tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/multi_objective_bo.ipynb
# currently it works only with categorical and desriptor method free, botorch feature to implement acqf_list_mixed needs to be
# implemented first https://github.com/pytorch/botorch/issues/1272
# main difference to the multiobjective strategies is that we have a randomized list of acqfs, this has to be bring into accordance
# with the other strategies
class BoTorchQparegoStrategy(BotorchBasicBoStrategy):
    name: str = "botorch.qparego"

    def _init_acqf(self) -> None:
        pass

    def _init_objective(self) -> None:
        pass

    def calc_acquisition(self, experiments: pd.DataFrame, combined: bool = False):
        raise ValueError("ACQF calc not implemented for qparego")

    def _init_domain(self) -> None:
        # first part of this is doubled with qehvi --> maybe create a common base class
        if len(self.domain.output_features.get_by_objective(excludes=None)) < 2:
            raise ValueError(
                "At least two output features has to be defined in the domain."
            )
        for feat in self.domain.output_features.get_by_objective(excludes=None):
            if isinstance(feat.objective, IdentityObjective) == False:
                raise ValueError(
                    "Only `MaximizeObjective` and `MinimizeObjective` supported."
                )
            if feat.objective.w != 1.0:
                raise ValueError(
                    "Only objective functions with weight 1 are supported."
                )
        if (
            len(self.domain.get_features(CategoricalInput)) > 0
            and self.categorical_method != CategoricalMethodEnum.FREE
        ):
            raise ValueError(
                "Only FREE optimization method for categoricals supported so far."
            )
        if (
            len(self.domain.get_features(CategoricalDescriptorInput)) > 0
            and self.descriptor_method != CategoricalMethodEnum.FREE
        ):
            raise ValueError(
                "Only FREE optimization method for Categorical with Descriptor supported so far."
            )

        super()._init_domain()
        return

    def _ask(self, candidate_count: int):
        assert candidate_count > 0, "candidate_count has to be larger than zero."

        acqf_list = []
        with torch.no_grad():
            clean_experiments = self.domain.preprocess_experiments_any_valid_output(
                self.experiments
            )
            transformed = self.transformer.transform(clean_experiments)
            train_x, _ = self.get_training_tensors(
                transformed,
                self.domain.output_features.get_keys_by_objective(excludes=None),
            )
            pred = self.model.posterior(train_x).mean

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        transformed = self.transformer.transform(clean_experiments)
        observed_x, _ = self.get_training_tensors(
            transformed,
            self.domain.output_features.get_keys_by_objective(excludes=None),
        )

        for i in range(candidate_count):
            ref_point_mask = torch.from_numpy(
                get_ref_point_mask(domain=self.domain)
            ).to(**tkwargs)
            weights = (
                sample_simplex(
                    len(
                        self.domain.output_features.get_keys_by_objective(excludes=None)
                    ),
                    **tkwargs
                ).squeeze()
                * ref_point_mask
            )
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )

            acqf = get_acquisition_function(
                acquisition_function_name="qNEI"
                if self.acqf == AcquisitionFunctionEnum.QNEI
                else "qEI",
                model=self.model,
                objective=objective,
                X_observed=observed_x,
                mc_samples=self.num_sobol_samples,
                qmc=True,
                prune_baseline=True,
            )
            acqf_list.append(acqf)

        # optimize

        candidates = optimize_acqf_list(
            acq_function_list=acqf_list,
            bounds=self.get_bounds(),
            num_restarts=self.num_restarts,
            raw_samples=self.num_raw_samples,
            equality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearEqualityConstraint
            ),
            inequality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearInequalityConstraint
            ),
            fixed_features=self.get_fixed_features(),
            options={"batch_limit": 5, "maxiter": 200},
        )

        preds = self.model.posterior(X=candidates[0]).mean.detach().numpy()
        stds = np.sqrt(self.model.posterior(X=candidates[0]).variance.detach().numpy())

        df_candidates = pd.DataFrame(
            data=np.nan,
            index=range(candidate_count),
            columns=self.input_feature_keys
            + [
                i + "_pred"
                for i in self.domain.output_features.get_keys_by_objective(
                    excludes=None
                )
            ]
            + [
                i + "_sd"
                for i in self.domain.output_features.get_keys_by_objective(
                    excludes=None
                )
            ]
            + [
                i + "_des"
                for i in self.domain.output_features.get_keys_by_objective(
                    excludes=None
                )
            ]
            # ["reward","acqf","strategy"]
        )

        for i, feat in enumerate(
            self.domain.output_features.get_by_objective(excludes=None)
        ):
            df_candidates[feat.key + "_pred"] = preds[:, i]
            df_candidates[feat.key + "_sd"] = stds[:, i]
            df_candidates[feat.key + "_des"] = feat.objective(preds[:, i])

        df_candidates[self.input_feature_keys] = candidates[0].detach().numpy()

        return self.transformer.inverse_transform(df_candidates)

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == NChooseKConstraint:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        if my_type not in [MaximizeObjective, MinimizeObjective]:
            return False
        return True
