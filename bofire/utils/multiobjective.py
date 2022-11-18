from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from bofire.domain import Domain
from bofire.domain.desirability_functions import (
    MaxIdentityDesirabilityFunction,
    MinIdentityDesirabilityFunction,
)
from bofire.domain.features import (
    ContinuousOutputFeature,
    ContinuousOutputFeature_woDesFunc,
    OutputFeature,
)


def get_ref_point_mask(domain:Domain, output_feature_keys:Optional[list]=None)->np.array:
    if output_feature_keys is None:
        output_feature_keys = domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])
    if len(output_feature_keys) < 2: raise ValueError("At least two output features have to be provided.")
    mask = []
    for key in output_feature_keys:
        feat = domain.get_feature(key)
        if isinstance(feat.desirability_function,MaxIdentityDesirabilityFunction):
            mask.append(1.)
        elif isinstance(feat.desirability_function,MinIdentityDesirabilityFunction):
            mask.append(-1.)
        else:
            raise ValueError("Only `MaxIdentityDesirabilityFunction` and `MinIdentityDesirabilityFunction` supported.")
    return np.array(mask)

def get_pareto_front(domain:Domain,experiments:pd.DataFrame, output_feature_keys:Optional[list]=None)->pd.DataFrame:   
    if output_feature_keys is None:
        output_feature_keys = domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])
    assert len(output_feature_keys) >= 2, "At least two output features have to be provided."
    df = domain.preprocess_experiments_all_valid_outputs(experiments, output_feature_keys)
    ref_point_mask = get_ref_point_mask(domain, output_feature_keys)
    pareto_mask = np.array(
        is_non_dominated(
            torch.from_numpy(
                df[output_feature_keys].values*ref_point_mask
            )
        )
    )
    return df.loc[pareto_mask]

def compute_hypervolume(domain:Domain, optimal_experiments:pd.DataFrame, ref_point: dict) -> float:
    ref_point_mask = get_ref_point_mask(domain)
    hv = Hypervolume(
        ref_point=torch.from_numpy(np.array([ref_point[feat] for feat in domain.get_feature_keys(ContinuousOutputFeature, excludes=ContinuousOutputFeature_woDesFunc)])*ref_point_mask))
    return hv.compute(torch.from_numpy(optimal_experiments[domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])].values*ref_point_mask))

def infer_ref_point(domain:Domain, experiments:pd.DataFrame,return_masked:bool = False):
    df = domain.preprocess_experiments_all_valid_outputs(experiments)
    mask = get_ref_point_mask(domain)
    ref_point_array = (df[domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])].values*mask).min(axis=0)
    if return_masked == False:
        ref_point_array/=mask
    return {feat: ref_point_array[i] for i, feat in enumerate(domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]))}
