from typing import Dict, List, Optional

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


def get_dim_subsets(d: int, active_dims: List[int], cat_dims: List[int]):
    def check_indices(d, indices):
        if len(set(indices)) != len(indices):
            raise ValueError("Elements of `indices` list must be unique!")
        if any([i > d - 1 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        if len(indices) > d:
            raise ValueError("Can provide at most `d` indices!")
        if any([i < 0 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        return indices

    if len(active_dims) == 0:
        raise ValueError("At least one active dim has to be provided!")

    # check validity of indices
    active_dims = check_indices(d, active_dims)
    cat_dims = check_indices(d, cat_dims)

    # compute subsets
    ord_dims = sorted(set(range(d)) - set(cat_dims))
    ord_active_dims = sorted(
        set(active_dims) - set(cat_dims)
    )  # includes also descriptors
    cat_active_dims = sorted([i for i in cat_dims if i in active_dims])
    return ord_dims, ord_active_dims, cat_active_dims


class SingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]
        # transform X
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        # transform from pandas to torch
        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        if tX.dim() == 2:
            batch_shape = torch.Size()
        else:
            batch_shape = torch.Size([tX.shape[0]])

        d = tX.shape[-1]

        cat_dims = []
        for feat in non_numerical_features:
            cat_dims += features2idx[feat]

        ord_dims, _, _ = get_dim_subsets(
            d=d, active_dims=list(range(d)), cat_dims=cat_dims
        )
        # first get the scaler
        # TODO use here the correct bounds
        if self.scaler == ScalerEnum.NORMALIZE:
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )

            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                batch_shape=batch_shape,
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims if len(ord_dims) != d else None,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError("Scaler enum not known.")

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=self.kernel.to_gpytorch(
                batch_shape=batch_shape, active_dims=list(range(d)), ard_num_dims=1
            ),
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=scaler,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
