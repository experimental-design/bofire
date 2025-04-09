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
from bofire.data_models.strategies.predictives.mobo import (
    AbsoluteMovingReferenceValue,
    ExplicitReferencePoint,
    FixedReferenceValue,
)
from bofire.utils.torch_tools import get_multiobjective_objective, tkwargs


def get_ref_point_mask(
    domain: Domain,
    output_feature_keys: Optional[list] = None,
) -> np.ndarray:
    """Method to get a mask for the reference points taking into account if we
    want to maximize or minimize an objective. In case it is maximize the value
    in the mask is 1, in case we want to minimize it is -1.

    Args:
        domain: Domain for which the mask should be generated.
        output_feature_keys: Name of output feature keys
            that should be considered in the mask. If `None` is provided, all keys
            belonging to output features with one of the following objectives will
            be considered: `MaximizeObjective`, `MinimizeObjective`,
            `CloseToTargetObjective`. Defaults to None.

    Returns:
        Array of ones for maximization and array of negative ones for
            minimization.

    """
    if output_feature_keys is None:
        output_feature_keys = domain.outputs.get_keys_by_objective(
            includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
    if len(output_feature_keys) < 2:
        raise ValueError("At least two output features have to be provided.")
    mask = []
    for key in output_feature_keys:
        feat = domain.outputs.get_by_key(key)
        if isinstance(feat.objective, MaximizeObjective):
            mask.append(1.0)
        elif isinstance(feat.objective, MinimizeObjective) or isinstance(
            feat.objective,
            CloseToTargetObjective,
        ):
            mask.append(-1.0)
        else:
            raise ValueError(
                "Only `MaximizeObjective` and `MinimizeObjective` supported",
            )
    return np.array(mask)


def get_pareto_front(
    domain: Domain,
    experiments: pd.DataFrame,
    output_feature_keys: Optional[list] = None,
) -> pd.DataFrame:
    """Method to compute the pareto optimal experiments for a given domain and
    experiments.

    Args:
        domain: Domain for which the pareto front should be computed.
        experiments: Experiments for which the pareto front should be computed.
        output_feature_keys: Key of output feature that should be considered
            in the pareto front. If `None` is provided, all keys
            belonging to output features with one of the following objectives will
            be considered: `MaximizeObjective`, `MinimizeObjective`,
            `CloseToTargetObjective`. Defaults to None.

    Returns:
        DataFrame with pareto optimal experiments.

    """
    if output_feature_keys is None:
        outputs = domain.outputs.get_by_objective(
            includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
    else:
        outputs = domain.outputs.get_by_keys(output_feature_keys)
    assert len(outputs) >= 2, "At least two output features have to be provided."
    output_feature_keys = [f.key for f in outputs]
    df = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments,
        output_feature_keys,
    )
    objective = get_multiobjective_objective(outputs=outputs, experiments=experiments)
    pareto_mask = np.array(
        is_non_dominated(
            objective(
                torch.from_numpy(df[output_feature_keys].values).to(**tkwargs),
                None,
            ),
        ),
    )
    return df.loc[pareto_mask]


def compute_hypervolume(
    domain: Domain,
    optimal_experiments: pd.DataFrame,
    ref_point: dict,
) -> float:
    """Method to compute the hypervolume for a given domain and pareto optimal experiments.

    Args:
        domain: Domain for which the hypervolume should be computed.
        optimal_experiments: Pareto optimal experiments for which the hypervolume
            should be computed.
        ref_point: Unmasked reference point for the hypervolume computation.
            Masking is happening inside the method.

    Returns:
        Hypervolume for the given domain and pareto optimal experiments.
    """
    outputs = domain.outputs.get_by_objective(
        includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
    )
    objective = get_multiobjective_objective(
        outputs=outputs,
        experiments=optimal_experiments,
    )
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
                    ],
                )
            ],
        ).to(**tkwargs)
        * ref_point_mask,
    )

    return hv.compute(
        objective(
            torch.from_numpy(
                optimal_experiments[
                    domain.outputs.get_keys_by_objective(
                        includes=[
                            MaximizeObjective,
                            MinimizeObjective,
                            CloseToTargetObjective,
                        ],
                    )
                ].values,  # type: ignore
            ).to(**tkwargs),
        ),
    )


def infer_ref_point(
    domain: Domain,
    experiments: pd.DataFrame,
    return_masked: bool = False,
    reference_point: Optional[ExplicitReferencePoint] = None,
) -> Dict[str, float]:
    """Method to infer the reference point for a given domain and experiments.

    Args:
        domain: Domain for which the reference point should be inferred.
        experiments: Experiments for which the reference point should be inferred.
        return_masked: If True, the masked reference point is returned. If False,
            the unmasked reference point is returned. Defaults to False.
        reference_point: Reference point to be used. If None is provided, the
            reference value is inferred by the worst values seen so far for
            every objective. Defaults to None.
    """
    outputs = domain.outputs.get_by_objective(
        includes=[MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
    )
    keys = outputs.get_keys()

    if reference_point is None:
        reference_point = ExplicitReferencePoint(
            values={
                key: AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)
                for key in keys
            }
        )

    objective = get_multiobjective_objective(outputs=outputs, experiments=experiments)

    df = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments,
        output_feature_keys=keys,
    )

    worst_values_array = (
        objective(torch.from_numpy(df[keys].values).to(**tkwargs), None)
        .numpy()
        .min(axis=0)
    )

    best_values_array = (
        objective(torch.from_numpy(df[keys].values).to(**tkwargs), None)
        .numpy()
        .max(axis=0)
    )
    # In the ref_point_array want masked values, which means that
    # maximization is assumed for everything, this is because we use
    # botorch objective for getting the best and worst values
    # that are passed to the reference values. Botorch always assumes
    # maximization.
    # In case of FixedReferenceValue, the unmasked values are stored in the data model,
    # this means the need to mask them to account for the botorch convention.
    # In case of FixedReferenceValue, we multiply with -1 in for `MinimizeObjective`
    # and `CloseToTargetObjective`.
    ref_point_array = np.array(
        [
            -reference_point.values[key].get_reference_value(
                best=best_values_array[i], worst=worst_values_array[i]
            )
            if isinstance(reference_point.values[key], FixedReferenceValue)
            and isinstance(
                outputs[i].objective, (MinimizeObjective, CloseToTargetObjective)
            )
            else reference_point.values[key].get_reference_value(
                best=best_values_array[i], worst=worst_values_array[i]
            )
            for i, key in enumerate(keys)
        ]
    )

    mask = get_ref_point_mask(domain)

    # here we unmask again by dividing by the mask
    if return_masked is False:
        ref_point_array /= mask
    return {feat: ref_point_array[i] for i, feat in enumerate(keys)}
