from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


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
            includes=[MaximizeObjective, MinimizeObjective]
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
    """
    Compute the Pareto front for the given set of experiments and output features.

    Args:
        domain (Domain): The domain object of the optimization problem.
        experiments (pd.DataFrame): A Pandas DataFrame containing the results of the experiments.
        output_feature_keys (Optional[list]): A list of strings representing the output features to use.
            If None, the output features are inferred from the domain using the MaximizeObjective and MinimizeObjective.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the Pareto front of the experiments.
    """

    if output_feature_keys is None:
        output_feature_keys = domain.outputs.get_keys_by_objective(
            includes=[MaximizeObjective, MinimizeObjective]
        )
    assert (
        len(output_feature_keys) >= 2
    ), "At least two output features have to be provided."
    df = domain.outputs.preprocess_experiments_all_valid_outputs(
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
    """
    Computes the hypervolume of the given set of experiments with respect to the reference point.

    Args:
        domain (Domain): The search domain object for the optimization problem.
        optimal_experiments (pd.DataFrame): A Pandas DataFrame containing the optimal results of the experiments.
        ref_point (dict): A dictionary representing the reference point in the output feature space.

    Returns:
        float: The hypervolume of the experiments with respect to the reference point.
    """

    ref_point_mask = get_ref_point_mask(domain)
    hv = Hypervolume(
        ref_point=torch.from_numpy(
            np.array(
                [
                    ref_point[feat]
                    for feat in domain.outputs.get_keys_by_objective(
                        includes=[MaximizeObjective, MinimizeObjective]
                    )
                ]
            )
            * ref_point_mask
        )
    )
    return hv.compute(
        torch.from_numpy(
            optimal_experiments[
                domain.outputs.get_keys_by_objective(
                    includes=[MaximizeObjective, MinimizeObjective]
                )
            ].values
            * ref_point_mask
        )
    )


def infer_ref_point(
    domain: Domain, experiments: pd.DataFrame, return_masked: bool = False
):
    """
    Infers the reference point for the given set of experiments in the output feature space.

    Args:
        domain (Domain): The search domain object for the optimization problem.
        experiments (pd.DataFrame): A Pandas DataFrame containing the results of the experiments.
        return_masked (bool): Whether to return the reference point in masked or unmasked form. Defaults to False.

    Returns:
        dict: A dictionary representing the reference point in the output feature space.
    """

    keys = domain.outputs.get_keys_by_objective(
        includes=[MaximizeObjective, MinimizeObjective]
    )
    df = domain.outputs.preprocess_experiments_all_valid_outputs(
        experiments, output_feature_keys=keys
    )
    mask = get_ref_point_mask(domain)
    ref_point_array = (df[keys].values * mask).min(axis=0)
    if return_masked is False:
        ref_point_array /= mask
    return {feat: ref_point_array[i] for i, feat in enumerate(keys)}
