import itertools
import warnings
from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import CategoricalInput, ContinuousInput


def get_confounding_matrix(
    inputs: Inputs,
    design: pd.DataFrame,
    powers: Optional[List[int]] = None,
    interactions: Optional[List[int]] = None,
):
    """Analyzes the confounding of a design and returns the confounding matrix.

    Only takes continuous features into account.

    Args:
        inputs (Inputs): Input features.
        design (pd.DataFrame): Design matrix.
        powers (List[int], optional): List of powers of the individual factors/features that should be considered.
            Integers has to be larger than 1. Defaults to [].
        interactions (List[int], optional): List with interaction levels to be considered.
            Integers has to be larger than 1. Defaults to [2].

    Returns:
        _type_: _description_
    """
    if len(inputs.get(CategoricalInput)) > 0:
        warnings.warn("Categorical input features will be ignored.")

    keys = inputs.get_keys(ContinuousInput)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_design = pd.DataFrame(
        data=scaler.fit_transform(design[keys]),
        columns=keys,
    )

    # add powers
    if powers is not None:
        for p in powers:
            assert p > 1, "Power has to be at least of degree two."
            for key in keys:
                scaled_design[f"{key}**{p}"] = scaled_design[key] ** p

    # add interactions
    if interactions is None:
        interactions = [2]

    for i in interactions:
        assert i > 1, "Interaction has to be at least of degree two."
        assert i < len(keys) + 1, f"Interaction has to be smaller than {len(keys)+1}."
        for combi in itertools.combinations(keys, i):
            scaled_design[":".join(combi)] = scaled_design[list(combi)].prod(axis=1)

    return scaled_design.corr()
