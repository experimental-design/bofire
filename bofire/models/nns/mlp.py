from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.input import InputStandardize, Normalize
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bofire.models.gps.gps import get_dim_subsets
from bofire.models.model import TrainableModel
from bofire.models.torch_models import BotorchModel
from bofire.utils.enum import CategoricalEncodingEnum, ScalerEnum
from bofire.utils.torch_tools import tkwargs


class RegressionDataSet(Dataset):
    """
    Prepare the dataset for regression
    """

    def __init__(self, X: Tensor, y: Tensor):
        self.X = X.to(**tkwargs)
        self.y = y.to(**tkwargs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_layer_sizes: Sequence = (100,),
        dropout: float = 0.0,
        activation: Literal["relu", "logistic", "tanh"] = "relu",
    ):
        super().__init__()
        if activation == "relu":
            f_activation = nn.ReLU
        elif activation == "logistic":
            f_activation = nn.Sigmoid
        elif activation == "tanh":
            f_activation = nn.Tanh
        else:
            raise ValueError(f"Activation {activation} not known.")
        layers = [
            nn.Linear(input_size, hidden_layer_sizes[0]).to(**tkwargs),
            f_activation(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if len(hidden_layer_sizes) > 1:
            for i in range(len(hidden_layer_sizes) - 1):
                layers += [
                    nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]).to(
                        **tkwargs
                    ),
                    f_activation(),
                ]
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_size).to(**tkwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _MLPEnsemble(EnsembleModel):
    def __init__(self, mlps: Sequence[MLP]):
        super().__init__()
        if len(mlps) == 0:
            raise ValueError("List of mlps is empty.")
        num_in_features = mlps[0].layers[0].in_features
        num_out_features = mlps[0].layers[-1].out_features
        for mlp in mlps:
            assert mlp.layers[0].in_features == num_in_features
            assert mlp.layers[-1].out_features == num_out_features
        self.mlps = mlps
        # put all models in eval mode
        for mlp in self.mlps:
            mlp.eval()

    def forward(self, X: Tensor):
        r"""Compute the model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x s x n x m`-dimensional output tensor where
            `s` is the size of the ensemble.
        """
        return torch.stack([mlp(X) for mlp in self.mlps], dim=-3)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self.mlps[0].layers[-1].out_features  # type: ignore


def fit_mlp(
    mlp: MLP,
    dataset: RegressionDataSet,
    batch_size: int = 10,
    n_epoches: int = 200,
    lr: float = 1e-4,
    shuffle: bool = True,
    weight_decay: float = 0.0,
):
    """Fit a MLP for regression to a dataset.

    Args:
        mlp (MLP): The MLP that should be fitted.
        dataset (RegressionDataSet): The data that should be fitted
        batch_size (int, optional): Batch size. Defaults to 10.
        n_epoches (int, optional): Number of training epoches. Defaults to 200.
        lr (float, optional): Initial learning rate. Defaults to 1e-4.
        shuffle (bool, optional): Whereas the batches should be shuffled. Defaults to True.
        weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 0.0 (no regularization).
    """
    mlp.train()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epoches):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get and prepare inputs
            inputs, targets = data
            if len(targets.shape) == 1:
                targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()


class MLPEnsemble(BotorchModel, TrainableModel):
    type: Literal["MLPEnsemble"] = "MLPEnsemble"
    size: int
    hidden_layer_sizes: Sequence = (100,)
    activation: Literal["relu", "logistic", "tanh"] = "relu"
    dropout: float = 0.0
    batch_size: int = 10
    n_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    subsample_fraction: float = 1.0
    shuffle: bool = True
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    model: Optional[_MLPEnsemble] = None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # TODO: this is very similar to the GPs, this has to be tidied up

        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]

        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        d = transformed_X.shape[-1]

        cat_dims = []
        for feat in non_numerical_features:
            cat_dims += features2idx[feat]

        ord_dims, _, _ = get_dim_subsets(
            d=d, active_dims=list(range(d)), cat_dims=cat_dims
        )

        if self.scaler == ScalerEnum.NORMALIZE:
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )
            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                batch_shape=torch.Size(),
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims if len(ord_dims) != d else None,
                batch_shape=torch.Size(),
            )
        else:
            raise ValueError("Scaler enum not known.")

        mlps = []
        subsample_size = round(self.subsample_fraction * X.shape[0])
        for i in range(self.size):
            # resample X and Y
            sample_idx = np.random.choice(X.shape[0], replace=True, size=subsample_size)
            tX = torch.from_numpy(transformed_X.values[sample_idx]).to(**tkwargs)
            ty = torch.from_numpy(Y.values[sample_idx]).to(**tkwargs)

            dataset = RegressionDataSet(
                X=scaler.transform(tX),
                y=ty,
            )
            mlp = MLP(
                input_size=transformed_X.shape[1],
                output_size=1,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                dropout=self.dropout,
            )
            fit_mlp(
                mlp=mlp,
                dataset=dataset,
                batch_size=self.batch_size,
                n_epoches=self.n_epochs,
                lr=self.lr,
                shuffle=self.shuffle,
                weight_decay=self.weight_decay,
            )
            mlps.append(mlp)
        self.model = _MLPEnsemble(mlps=mlps)
        self.model.input_transform = scaler
