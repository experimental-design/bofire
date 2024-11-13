import base64
import io
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from torch import Tensor

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import RandomForestSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


class _RandomForest(EnsembleModel):
    """Botorch wrapper around the sklearn RandomForestRegressor.
    Predictions of the individual trees are interpreted as uncertainty.
    """

    def __init__(
        self,
        rf: RandomForestRegressor,
        output_scaler: Optional[OutcomeTransform] = None,
    ):
        """Constructs the model.

        Args:
            rf (RandomForestRegressor): Fitted sklearn random forest regressor.

        """
        super().__init__()
        if not isinstance(rf, RandomForestRegressor):
            raise ValueError("`rf` is not a sklearn RandomForestRegressor.")
        check_is_fitted(rf)
        self._rf = rf
        if output_scaler is not None:
            self.outcome_transform = output_scaler

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
                    estimator.predict(nX[i]).reshape((nX[i].shape[0], 1)),
                )
            preds.append(np.stack(batch_preds, axis=0))
        preds = np.stack(preds, axis=0)
        if X.ndim == 3:  # we have a batch dim
            return torch.from_numpy(preds).to(**tkwargs)
        # we have no batch dim
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
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
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

        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        tX = (
            scaler.transform(torch.from_numpy(transformed_X.values)).numpy()
            if scaler is not None
            else transformed_X.values
        )

        if self.output_scaler == ScalerEnum.STANDARDIZE:
            output_scaler = Standardize(m=Y.shape[-1])
            ty = torch.from_numpy(Y.values).to(**tkwargs)
            ty = output_scaler(ty)[0].numpy()
        else:
            output_scaler = None
            ty = Y.values

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        rf.fit(X=tX, y=ty.ravel())

        self.model = _RandomForest(rf=rf, output_scaler=output_scaler)
        if scaler is not None:
            self.model.input_transform = scaler

    def _dumps(self) -> str:
        """Dumps the random forest to a string via pickle as this is not directly json serializable."""
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str):
        """Loads the actual random forest from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, weights_only=False)
