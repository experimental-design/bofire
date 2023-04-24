import codecs
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.ensemble import EnsembleModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from torch import Tensor

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import RandomForestSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
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
        if not isinstance(rf, RandomForestRegressor):
            raise ValueError("`rf` is not a sklearn RandomForestRegressor.")
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


class RandomForestSurrogate(BotorchSurrogate, TrainableSurrogate):
    """BoFire Random Forest model.

    The same hyperparameters are available as for the wrapped sklearn RandomForestRegreesor.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_estimators = data_model.n_estimators
        self.criterion = data_model.criterion
        self.max_depth = data_model.max_depth
        self.min_samples_split = data_model.min_samples_split
        self.min_samples_leaf = data_model.min_samples_leaf
        self.min_weight_fraction_leaf = data_model.min_weight_fraction_leaf
        self.max_features = data_model.max_features
        self.max_leaf_nodes = data_model.max_leaf_nodes
        self.min_impurity_decrease = data_model.min_impurity_decrease
        self.bootstrap = data_model.bootstrap
        self.oob_score = data_model.oob_score
        self.random_state = data_model.random_state
        self.ccp_alpha = data_model.ccp_alpha
        self.max_samples = data_model.max_samples
        super().__init__(data_model=data_model, **kwargs)

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    model: Optional[_RandomForest] = None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Fit the Random Forest model.

        Args:
            X (pd.DataFrame): Dataframe with X values.
            Y (pd.DataFrame): Dataframe with Y values.
        """
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
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

    def _dumps(self) -> str:
        """Dumps the random forest to a string via pickle as this is not directly json serializable."""
        return codecs.encode(pickle.dumps(self.model._rf), "base64").decode()  # type: ignore

    def loads(self, data: str):
        """Loads the actual random forest from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        self.model = _RandomForest(
            rf=pickle.loads(codecs.decode(data.encode(), "base64"))
        )
