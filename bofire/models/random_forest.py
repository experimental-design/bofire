from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from botorch.models.ensemble import EnsembleModel
from pydantic.types import NonNegativeFloat
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from torch import Tensor

from bofire.models.model import TrainableModel
from bofire.models.torch_models import BotorchModel
from bofire.utils.enum import OutputFilteringEnum
from bofire.utils.torch_tools import tkwargs


class _RandomForest(EnsembleModel):
    """Botorch wrapper around the sklearn RandomForestRegressor.
    Predictions of the individual trees are interpreted as uncertainty.
    """

    def __init__(self, rf: RandomForestRegressor):
        """Constructs the model.

        Args:
            rf (RandomForestRegressor): Fitted sklearn random forest regressor.
        """
        super().__init__()
        check_is_fitted(rf)
        self._rf = rf

    def forward(self, X: Tensor):
        r"""Compute the model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x s x n x m`-dimensional output tensor where
            `s` is the size of the ensemble.
        """
        # we transform to numpy
        nX = X.detach().numpy()
        # we need to check if we have a batch dimension
        if len(X.shape) != 3:
            # now we add the q-batch dimension
            nX = nX.reshape((1, *nX.shape))
        # loop over batches
        preds = []
        for i in range(nX.shape[0]):
            batch_preds = []
            # loop over estimators
            for estimator in self._rf.estimators_:
                batch_preds.append(
                    estimator.predict(nX[i]).reshape((nX[i].shape[0], 1))
                )
            preds.append(np.stack(batch_preds, axis=0))
        preds = np.stack(preds, axis=0)
        if X.ndim == 3:  # we have a batch dim
            return torch.from_numpy(preds).to(**tkwargs)
        else:  # we have no batch dim
            return torch.from_numpy(preds).to(**tkwargs).squeeze(dim=0)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1


class RandomForest(BotorchModel, TrainableModel):
    """BoFire Random Forest model.

    The same hyperparameters are available as for the wrapped sklearn RandomForestRegreesor.
    """

    # hyperparams passed down to `RandomForestRegressor`
    n_estimators: int = 100
    criterion: str = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float] = 1
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    random_state: Optional[int] = None
    ccp_alpha: NonNegativeFloat = 0.0
    max_samples: Optional[Union[int, float]] = None

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    model: Optional[_RandomForest] = None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Fit the Random Forest model.

        Args:
            X (pd.DataFrame): Dataframe with X values.
            Y (pd.DataFrame): Dataframe with Y values.
        """
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,  # type: ignore
            min_samples_leaf=self.min_samples_leaf,  # type: ignore
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,  # type: ignore
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        rf.fit(X=transformed_X.values, y=Y.values.ravel())
        self.model = _RandomForest(rf=rf)
