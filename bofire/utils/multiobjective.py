from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.utils.torch_tools import get_multiobjective_objective, tkwargs


def get_ref_point_mask(
    domain: Domain, output_feature_keys: Optional[list] = None
) -> np.ndarray:
    """Method to get a mask for the reference points taking into account if we
    want to maximize or minimize an objective. In case it is maximize the value
    in the mask is 1, in case we want to minimize it is -1.

    Args:
        domain (Domain): Domain for which the mask should be generated.
        output_feature_keys (Optional[list], optional): Name of output feature keys
            that should be considered in the mask. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    if output_feature_keys is None:
        output_feature_keys = domain.outputs.get_keys_by_objective(
            includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective]
        )
    if len(output_feature_keys) < 2:
        raise ValueError("At least two output features have to be provided.")
    mask = []
    for key in output_feature_keys:
        feat = domain.get_feature(key)
        if isinstance(feat.objective, MaximizeObjective):  # type: ignore
            mask.append(1.0)
        elif isinstance(feat.objective, MinimizeObjective):  # type: ignore
            mask.append(-1.0)
        elif isinstance(feat.objective, CloseToTargetObjective):  # type: ignore
            mask.append(-1.0)
        else:
            raise ValueError(
                "Only `MaximizeObjective` and `MinimizeObjective` supported"
            )
    return np.array(mask)


def get_pareto_front(
    domain: Domain,
    experiments: pd.DataFrame,
    output_feature_keys: Optional[list] = None,
) -> pd.DataFrame:
    if output_feature_keys is None:
        outputs = domain.outputs.get_by_objective(
            includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective]
        )
    else:
        outputs = domain.outputs.get_by_keys(output_feature_keys)
    assert len(outputs) >= 2, "At least two output features have to be provided."
    output_feature_keys = [f.key for f in outputs]
    df = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments, output_feature_keys
    )
    objective = get_multiobjective_objective(outputs=outputs)  # type: ignore
    pareto_mask = np.array(
        is_non_dominated(
            objective(
                torch.from_numpy(df[output_feature_keys].values).to(**tkwargs), None
            )
        )
    )
    return df.loc[pareto_mask]


def compute_hypervolume(
    domain: Domain, optimal_experiments: pd.DataFrame, ref_point: dict
) -> float:
    outputs = domain.outputs.get_by_objective(
        includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective]
    )
    objective = get_multiobjective_objective(outputs=outputs)
    ref_point_mask = torch.from_numpy(get_ref_point_mask(domain)).to(**tkwargs)
    hv = Hypervolume(
        ref_point=torch.tensor(
            [
                ref_point[feat]
                for feat in domain.outputs.get_keys_by_objective(
                    includes=[
                        MaximizeObjective,
                        MinimizeObjective,
                        CloseToTargetObjective,
                    ]
                )
            ]
        ).to(**tkwargs)
        * ref_point_mask
    )

    return hv.compute(
        objective(  # type: ignore
            torch.from_numpy(
                optimal_experiments[
                    domain.outputs.get_keys_by_objective(
                        includes=[
                            MaximizeObjective,
                            MinimizeObjective,
                            CloseToTargetObjective,
                        ]
                    )
                ].values
            ).to(**tkwargs)
        )
    )


def infer_ref_point(
    domain: Domain, experiments: pd.DataFrame, return_masked: bool = False
) -> Dict[str, float]:
    outputs = domain.outputs.get_by_objective(
        includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective]
    )
    keys = [f.key for f in outputs]
    objective = get_multiobjective_objective(outputs=outputs)

    df = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments, output_feature_keys=keys
    )

    ref_point_array = (
        objective(torch.from_numpy(df[keys].values).to(**tkwargs), None)
        .numpy()
        .min(axis=0)
    )

    mask = get_ref_point_mask(domain)
    # ref_point_array = (df[keys].values * mask).min(axis=0)
    if return_masked is False:
        ref_point_array /= mask
    return {feat: ref_point_array[i] for i, feat in enumerate(keys)}
