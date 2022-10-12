from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from enum import Enum
from everest.domain.constraints import ConcurrencyConstraint, Constraint
from everest.domain.desirability_functions import (
    IdentityDesirabilityFunction, MaxIdentityDesirabilityFunction,
    MinIdentityDesirabilityFunction)
from everest.domain.features import (ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     InputFeature, 
                                     OutputFeature, 
                                     CategoricalDescriptorInputFeature,
                                     CategoricalInputFeature)
from everest.domain.constraints import (LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.base import BotorchBasicBoStrategy
from everest.strategies.strategy import Strategy, RandomStrategy
from everest.utils.multiobjective import get_ref_point_mask
from everest.strategies.strategy import (CategoricalMethodEnum,
                                         RandomStrategy)
from pydantic import validator
from pydantic.types import conint, conlist

from botorch.optim.optimize import optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.utils import get_acquisition_function



class AcquisitionFunctionEnum(Enum):
    QNEI = "QNEI"
    QEI = "QEI"

# this implementation follows this tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/multi_objective_bo.ipynb
# currently it works only with categorical and desriptor method free, botorch feature to implement acqf_list_mixed needs to be 
# implemented first https://github.com/pytorch/botorch/issues/1272
# main difference to the multiobjective strategies is that we have a randomized list of acqfs, this has to be bring into accordance
# with the other strategies
class BoTorchQparegoStrategy(BotorchBasicBoStrategy):

    base_acquisition_function: AcquisitionFunctionEnum = AcquisitionFunctionEnum.QNEI

    def _init_acqf(self) -> None:
        pass

    def _init_objective(self) -> None:
        pass

    def calc_acquisition(self, experiments: pd.DataFrame, combined: bool = False):
        raise ValueError("ACQF calc not implemented for qparego")

    def _init_domain(self) -> None:
        # first part of this is doubled with qehvi --> maybe create a common base class
        if len(self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])) < 2:
            raise ValueError(
                "At least two output features has to be defined in the domain."
            )
        for feat in self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]):
            if (
                isinstance(feat.desirability_function, IdentityDesirabilityFunction)
                == False
            ):
                raise ValueError(
                    "Only `MaxIdentityDesirabilityFunction` and `MinIdentityDesirabilityFunction` supported."
                )
            if feat.desirability_function.w != 1.0:
                raise ValueError("Only desirability functions with weight 1 are supported.")
        if len(self.domain.get_features(CategoricalInputFeature))>0 and self.categorical_method != CategoricalMethodEnum.FREE:
            raise ValueError("Only FREE optimization method for categoricals supported so far.")
        if len(self.domain.get_features(CategoricalDescriptorInputFeature))>0 and self.descriptor_method != CategoricalMethodEnum.FREE:
            raise ValueError("Only FREE optimization method for Categorical with Descriptor supported so far.")
        super()._init_domain()
        return

    def _ask(self, candidate_count: int):
        assert candidate_count > 0, "candidate_count has to be larger than zero."

        acqf_list = []
        with torch.no_grad():
            clean_experiments = self.domain.preprocess_experiments_any_valid_output(self.experiments)
            transformed = self.transformer.transform(clean_experiments)
            train_x, _ = self.get_training_tensors(transformed, self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]))
            pred = self.model.posterior(train_x).mean

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        transformed = self.transformer.transform(clean_experiments)
        observed_x, _ = self.get_training_tensors(transformed, self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]))

        for i in range(candidate_count):
            ref_point_mask = torch.from_numpy(get_ref_point_mask(domain=self.domain)).to(**tkwargs)
            weights = sample_simplex(len(self.domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])), **tkwargs).squeeze()*ref_point_mask
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            

            acqf = get_acquisition_function(
                acquisition_function_name="qNEI" if self.base_acquisition_function == AcquisitionFunctionEnum.QNEI else "qEI",
                model = self.model,
                objective = objective,
                X_observed = observed_x,
                mc_samples = self.num_sobol_samples,
                qmc = True, 
                prune_baseline = True
            )
            acqf_list.append(acqf)
        
        # optimize

        candidates = optimize_acqf_list(
            acq_function_list = acqf_list,
            bounds = self.get_bounds(),
            num_restarts = self.num_restarts,
            raw_samples = self.num_raw_samples,
            equality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearEqualityConstraint
                ),
            inequality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearInequalityConstraint
                ),
            fixed_features=self.get_fixed_features(),
            options={"batch_limit": 5, "maxiter": 200},
        )

        preds = self.model.posterior(X=candidates[0]).mean.detach().numpy()
        stds = np.sqrt(
            self.model.posterior(X=candidates[0]).variance.detach().numpy()
        )

        df_candidates = pd.DataFrame(
            data=np.nan,
            index=range(candidate_count),
            columns=self.input_feature_keys
            + [i + "_pred" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            + [i + "_sd" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            + [i + "_des" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            # ["reward","acqf","strategy"]
        )

        for i, feat in enumerate(self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])):
            df_candidates[feat.key + "_pred"] = preds[:, i]
            df_candidates[feat.key + "_sd"] = stds[:, i]
            df_candidates[feat.key + "_des"] = feat.desirability_function(preds[:, i])

        df_candidates[self.input_feature_keys] = candidates[0].detach().numpy()

        configs = self.get_candidate_log(candidates)
        return self.transformer.inverse_transform(df_candidates), configs

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == ConcurrencyConstraint:
            return False
        return True
