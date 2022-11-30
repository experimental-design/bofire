from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from bofire.domain import Domain
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective


def get_ref_point_mask(
    domain: Domain, output_feature_keys: Optional[list] = None
) -> np.array:
    if output_feature_keys is None:
        output_feature_keys = domain.output_features.get_keys_by_objective(
            excludes=None
        )
    if len(output_feature_keys) < 2:
        raise ValueError("At least two output features have to be provided.")
    mask = []
    for key in output_feature_keys:
        feat = domain.get_feature(key)
        if isinstance(feat.objective, MaximizeObjective):
            mask.append(1.0)
        elif isinstance(feat.objective, MinimizeObjective):
            mask.append(-1.0)
        else:
            raise ValueError(
                "Only `MaximizeObjective` and `MinimizeObjective` supported."
            )
    return np.array(mask)


def get_pareto_front(
    domain: Domain,
    experiments: pd.DataFrame,
    output_feature_keys: Optional[list] = None,
) -> pd.DataFrame:
    if output_feature_keys is None:
        output_feature_keys = domain.output_features.get_keys_by_objective(
            excludes=None
        )
    assert (
        len(output_feature_keys) >= 2
    ), "At least two output features have to be provided."
    df = domain.preprocess_experiments_all_valid_outputs(
        experiments, output_feature_keys
    )
    ref_point_mask = get_ref_point_mask(domain, output_feature_keys)
    pareto_mask = np.array(
        is_non_dominated(
            torch.from_numpy(df[output_feature_keys].values * ref_point_mask)
        )
    )
    return df.loc[pareto_mask]


def compute_hypervolume(
    domain: Domain, optimal_experiments: pd.DataFrame, ref_point: dict
) -> float:
    ref_point_mask = get_ref_point_mask(domain)
    hv = Hypervolume(
        ref_point=torch.from_numpy(
            np.array(
                [
                    ref_point[feat]
                    for feat in domain.output_features.get_keys_by_objective(
                        excludes=None
                    )
                ]
            )
            * ref_point_mask
        )
    )
    return hv.compute(
        torch.from_numpy(
            optimal_experiments[
                domain.output_features.get_keys_by_objective(excludes=None)
            ].values
            * ref_point_mask
        )
    )


def infer_ref_point(
    domain: Domain, experiments: pd.DataFrame, return_masked: bool = False
):
    df = domain.preprocess_experiments_all_valid_outputs(experiments)
    mask = get_ref_point_mask(domain)
    ref_point_array = (
        df[domain.output_features.get_keys_by_objective(excludes=None)].values * mask
    ).min(axis=0)
    if return_masked == False:
        ref_point_array /= mask
    return {
        feat: ref_point_array[i]
        for i, feat in enumerate(
            domain.output_features.get_keys_by_objective(excludes=None)
        )
    }
