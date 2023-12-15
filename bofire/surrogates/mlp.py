from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import MLPEnsemble as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
from bofire.surrogates.trainable import TrainableSurrogate
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
    def __init__(
        self, mlps: Sequence[MLP], output_scaler: Optional[OutcomeTransform] = None
    ):
        super().__init__()
        if len(mlps) == 0:
            raise ValueError("List of mlps is empty.")
        num_in_features = mlps[0].layers[0].in_features
        num_out_features = mlps[0].layers[-1].out_features
        for mlp in mlps:
            assert mlp.layers[0].in_features == num_in_features
            assert mlp.layers[-1].out_features == num_out_features
        self.mlps = mlps
        if output_scaler is not None:
            self.outcome_transform = output_scaler
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
        for data in train_loader:
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


class MLPEnsemble(BotorchSurrogate, TrainableSurrogate):
    def __init__(self, data_model: DataModel, **kwargs):
        self.n_estimators = data_model.n_estimators
        self.hidden_layer_sizes = data_model.hidden_layer_sizes
        self.activation = data_model.activation
        self.dropout = data_model.dropout
        self.batch_size = data_model.batch_size
        self.n_epochs = data_model.n_epochs
        self.lr = data_model.lr
        self.weight_decay = data_model.weight_decay
        self.subsample_fraction = data_model.subsample_fraction
        self.shuffle = data_model.shuffle
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model, **kwargs)

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    model: Optional[_MLPEnsemble] = None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        if self.output_scaler == ScalerEnum.STANDARDIZE:
            output_scaler = Standardize(m=Y.shape[-1])
        else:
            output_scaler = None

        mlps = []
        subsample_size = round(self.subsample_fraction * X.shape[0])
        for _ in range(self.n_estimators):
            # resample X and Y
            sample_idx = np.random.choice(X.shape[0], replace=True, size=subsample_size)
            tX = torch.from_numpy(transformed_X.values[sample_idx]).to(**tkwargs)
            ty = torch.from_numpy(Y.values[sample_idx]).to(**tkwargs)

            dataset = RegressionDataSet(
                X=scaler.transform(tX) if scaler is not None else tX,
                y=output_scaler(ty)[0] if output_scaler is not None else ty,
            )
            mlp = MLP(
                input_size=transformed_X.shape[1],
                output_size=1,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,  # type: ignore
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
        self.model = _MLPEnsemble(mlps, output_scaler=output_scaler)
        if scaler is not None:
            self.model.input_transform = scaler
