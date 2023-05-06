from typing import Literal, Union

import numpy as np
import pandas as pd
from pydantic import root_validator

from bofire.data_models.molfeatures.molfeatures import MolFeatures


class Fingerprints(MolFeatures):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    type: Literal["Fingerprints"] = "Fingerprints"
    bond_radius: int = 3
    n_bits: int = 2048

    def __call__(self, values: pd.Series,
        ) -> pd.DataFrame:
            # validate it
            data = smiles2fingerprints(
                values.to_list(), bond_radius=bond_radius, n_bits=n_bits
            )
            return pd.DataFrame(
                data=data,
                columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
            )


# class MaximizeObjective(IdentityObjective):
#     """Child class from the identity function without modifications, since the parent class is already defined as maximization
#
#     Attributes:
#         w (float): float between zero and one for weighting the objective
#         lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
#         upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
#     """
#
#     type: Literal["MaximizeObjective"] = "MaximizeObjective"
#
#
# class MinimizeObjective(IdentityObjective):
#     """Class returning the negative identity as reward.
#
#     Attributes:
#         w (float): float between zero and one for weighting the objective
#         lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
#         upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
#     """
#
#     type: Literal["MinimizeObjective"] = "MinimizeObjective"
#
#     def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
#         """The call function returning a reward for passed x values
#
#         Args:
#             x (np.ndarray): An array of x values
#
#         Returns:
#             np.ndarray: The negative identity as reward, might be normalized to the passed lower and upper bounds
#         """
#         return -1.0 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
