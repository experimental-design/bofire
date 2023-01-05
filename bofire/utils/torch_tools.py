from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import InputFeature

tkwargs = {
    "dtype": torch.double,
    "device": "cpu",
}


def get_linear_constraints(
    domain: Domain,
    constraint: Union[LinearEqualityConstraint, LinearInequalityConstraint],
    unit_scaled: bool = False,
) -> List[Tuple[Tensor, Tensor, float]]:
    """Converts linear constraints to the form required by BoTorch.

    Args:
        domain (Domain): Optimization problem definition.
        constraint (Union[LinearEqualityConstraint, LinearInequalityConstraint]): Type of constraint that should be converted.
        unit_scaled (bool, optional): If True, transforms constraints by assuming that the bound for the continuous features are [0,1]. Defaults to False.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of tuples, each tuple consists of a tensor with the feature indices, coefficients and a float for the rhs.
    """
    constraints = []
    for c in domain.cnstrs.get(constraint):
        indices = []
        coefficients = []
        lower = []
        upper = []
        rhs = 0.0
        for i, featkey in enumerate(c.features):  # type: ignore
            idx = domain.get_feature_keys(InputFeature).index(featkey)
            feat = domain.get_feature(featkey)
            if feat.is_fixed():  # type: ignore
                rhs -= feat.fixed_value()[0] * c.coefficients[i]  # type: ignore
            else:
                lower.append(feat.lower_bound)  # type: ignore
                upper.append(feat.upper_bound)  # type: ignore
                indices.append(idx)
                coefficients.append(
                    c.coefficients[i]  # type: ignore
                )  # if unit_scaled == False else c_scaled.coefficients[i])
        if unit_scaled:
            lower = np.array(lower)
            upper = np.array(upper)
            s = upper - lower
            scaled_coefficients = s * np.array(coefficients)
            constraints.append(
                (
                    torch.tensor(indices),
                    -torch.tensor(scaled_coefficients).to(**tkwargs),
                    -(rhs + c.rhs - np.sum(np.array(coefficients) * lower)),  # type: ignore
                )
            )
        else:
            constraints.append(
                (
                    torch.tensor(indices),
                    -torch.tensor(coefficients).to(**tkwargs),
                    -(rhs + c.rhs),  # type: ignore
                )
            )
    return constraints


# this is copied from https://github.com/pytorch/botorch/pull/1534, will be removed as soon as it is in botorch
# official


class OneHotToNumeric(InputTransform, Module):
    r"""Transform categorical parameters from a one-hot to a numeric representation.
    This assumes that the categoricals are the trailing dimensions.
    """

    def __init__(
        self,
        dim: int,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        r"""Initialize.
        Args:
            dim: The dimension of the one-hot-encoded input.
            categorical_features: A dictionary mapping the starting index of each
                categorical feature to its cardinality. This assumes that categoricals
                are one-hot encoded.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        Returns:
            A `batch_shape x n x d'`-dim tensor of where the one-hot encoded
            categoricals are transformed to integer representation.
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        categorical_features = categorical_features or {}
        # sort by starting index
        self.categorical_features = OrderedDict(
            sorted(categorical_features.items(), key=lambda x: x[0])
        )
        if len(self.categorical_features) > 0:
            self.categorical_start_idx = min(self.categorical_features.keys())
            # check that the trailing dimensions are categoricals
            end = self.categorical_start_idx
            err_msg = (
                f"{self.__class__.__name__} requires that the categorical "
                "parameters are the rightmost elements."
            )
            for start, card in self.categorical_features.items():
                # the end of one one-hot representation should be followed
                # by the start of the next
                if end != start:
                    raise ValueError(err_msg)
                end = start + card
            if end != dim:
                # check end
                raise ValueError(err_msg)
            # the numeric representation dimension is the total number of parameters
            # (continuous, integer, and categorical)
            self.numeric_dim = self.categorical_start_idx + len(categorical_features)

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the categorical inputs into integer representation.
        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.
        Returns:
            A `batch_shape x n x d'`-dim tensor of where the one-hot encoded
            categoricals are transformed to integer representation.
        """
        if len(self.categorical_features) > 0:
            X_numeric = X[..., : self.numeric_dim].clone()
            idx = self.categorical_start_idx
            for start, card in self.categorical_features.items():
                X_numeric[..., idx] = X[..., start : start + card].argmax(dim=-1)
                idx += 1
            return X_numeric
        return X

    def untransform(self, X: Tensor) -> Tensor:
        r"""Transform the categoricals from integer representation to one-hot.
        Args:
            X: A `batch_shape x n x d'`-dim tensor of transformed inputs, where
                the categoricals are represented as integers.
        Returns:
            A `batch_shape x n x d`-dim tensor of inputs, where the categoricals
            have been transformed to one-hot representation.
        """
        if len(self.categorical_features) > 0:
            self.numeric_dim
            one_hot_categoricals = [
                # note that self.categorical_features is sorted by the starting index
                # in one-hot representation
                one_hot(
                    X[..., idx - len(self.categorical_features)].long(),
                    num_classes=cardinality,
                )
                for idx, cardinality in enumerate(self.categorical_features.values())
            ]
            X = torch.cat(
                [
                    X[..., : self.categorical_start_idx],
                    *one_hot_categoricals,
                ],
                dim=-1,
            )
        return X
