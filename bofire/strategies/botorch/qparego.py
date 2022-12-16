import itertools
from typing import Type

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
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
    BotorchConstrainedObjective,
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.utils.enum import (
    AcquisitionFunctionEnum,
    CategoricalMethodEnum,
    DescriptorMethodEnum,
)
from bofire.utils.multiobjective import get_ref_point_mask
from bofire.utils.torch_tools import get_linear_constraints, tkwargs


# this implementation follows this tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/multi_objective_bo.ipynb
# currently it works only with categorical and desriptor method free, botorch feature to implement acqf_list_mixed needs to be
# implemented first https://github.com/pytorch/botorch/issues/1272
# main difference to the multiobjective strategies is that we have a randomized list of acqfs, this has to be bring into accordance
# with the other strategies
class BoTorchQparegoStrategy(BotorchBasicBoStrategy):
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
            if feat.objective.w != 1.0:  # type: ignore
                raise ValueError(
                    "Only objective functions with weight 1 are supported."
                )
        if (len(self.domain.get_features(CategoricalInput)) > 0) and (
            self.categorical_method != CategoricalMethodEnum.FREE
        ):
            raise ValueError(
                "Only FREE optimization method for categoricals supported so far."
            )
        if (len(self.domain.get_features(CategoricalDescriptorInput)) > 0) and (
            self.descriptor_method != DescriptorMethodEnum.FREE
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
            transformed = self.transformer.transform(clean_experiments)  # type: ignore
            train_x, _ = self.get_training_tensors(
                transformed,
                self.domain.output_features.get_keys_by_objective(excludes=None),  # type: ignore
            )
            pred = self.model.posterior(train_x).mean  # type: ignore

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(
            self.experiments
        )
        transformed = self.transformer.transform(clean_experiments)  # type: ignore
        observed_x, _ = self.get_training_tensors(
            transformed,
            self.domain.output_features.get_keys_by_objective(excludes=None),  # type: ignore
        )

        for i in range(candidate_count):
            ref_point_mask = torch.from_numpy(
                get_ref_point_mask(domain=self.domain)
            ).to(**tkwargs)
            weights = (
                sample_simplex(
                    len(
                        self.domain.output_features.get_keys_by_objective(
                            includes=[MaximizeObjective, MinimizeObjective]
                        )
                    ),
                    **tkwargs
                ).squeeze()
                * ref_point_mask
            )

            index_constraints = [
                idx
                for idx, feat in enumerate(
                    self.domain.output_features.get_by_objective().features
                )
                if isinstance(
                    feat.objective,
                    BotorchConstrainedObjective,
                )
            ]

            index_objectives = [
                idx
                for idx, feat in enumerate(
                    self.domain.output_features.get_by_objective().features
                )
                if not isinstance(
                    feat.objective,
                    BotorchConstrainedObjective,
                )
            ]

            scalarization = get_chebyshev_scalarization(
                weights=weights, Y=pred[..., index_objectives]
            )

            if len(index_constraints) > 0:
                objective = self.get_constrained_mc_objective(
                    index_objectives, index_constraints, scalarization
                )
            else:
                objective = GenericMCObjective(scalarization)

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
                domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
            ),
            inequality_constraints=get_linear_constraints(
                domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
            ),
            fixed_features=self.get_fixed_features(),
            options={"batch_limit": 5, "maxiter": 200},
        )

        preds = self.model.posterior(X=candidates[0]).mean.detach().numpy()  # type: ignore
        stds = np.sqrt(self.model.posterior(X=candidates[0]).variance.detach().numpy())  # type: ignore

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
            df_candidates[feat.key + "_des"] = feat.objective(preds[:, i])  # type: ignore

        df_candidates[self.input_feature_keys] = candidates[0].detach().numpy()

        return self.transformer.inverse_transform(df_candidates)  # type: ignore

    def get_constrained_mc_objective(
        self, index_objectives, index_constraints, scalarization
    ):
        def objective(Z):
            return scalarization(Z[..., index_objectives])

        constrained_objective = ConstrainedMCObjective(
            objective=objective,
            constraints=list(
                itertools.chain(
                    *[
                        self.domain.output_features.get_by_objective(excludes=None)
                        .features[idx]
                        .objective.to_constraints(idx)
                        for idx in index_constraints
                    ]
                )
            ),
        )
        return constrained_objective

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

        return True
