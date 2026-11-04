import base64
import io
from typing import Literal, Optional

import numpy as np
import torch
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from typing_extensions import Self

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import RandomForestSurrogate as DataModel
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate
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


class RandomForestSurrogate(TrainableBotorchSurrogate):
    """BoFire Random Forest model.

    The same hyperparameters are available as for the wrapped sklearn RandomForestRegreesor.
    """

    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
        scaler: AnyScaler = Normalize(),  # noqa: B008 - copied by make_surrogate
        output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE,
        n_estimators: int = 100,
        criterion: Literal[
            "squared_error", "absolute_error", "friedman_mse", "poisson"
        ] = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: int | float | Literal["auto", "sqrt", "log2"] = 1.0,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,
    ) -> Self:
        """Create a runtime random forest using validated data-model settings.

        Args:
            inputs: Input features.
            outputs: Output features.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            output_scaler: Output scaling method.
            n_estimators: Number of trees.
            criterion: Split quality criterion.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum sample count or fraction needed to split.
            min_samples_leaf: Minimum sample count or fraction in a leaf.
            min_weight_fraction_leaf: Minimum weighted sample fraction in a leaf.
            max_features: Number or fraction of features considered per split.
            max_leaf_nodes: Maximum leaf count per tree.
            min_impurity_decrease: Minimum impurity decrease needed to split.
            bootstrap: Whether trees use bootstrap samples.
            oob_score: Whether to compute an out-of-bag score.
            random_state: Random seed for the forest.
            ccp_alpha: Cost-complexity pruning parameter.
            max_samples: Sample count or fraction drawn for each bootstrap tree.

        Defaults mirror the data model and are copied for each call. Call ``fit`` to train.
        """
        return make_surrogate(cls, DataModel, locals())

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
        self.random_state = data_model.random_state
        self.ccp_alpha = data_model.ccp_alpha
        self.max_samples = data_model.max_samples
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    model: Optional[_RandomForest] = None

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ) -> None:
        # we have to apply the botorch transforms before fitting
        if input_transform is not None:
            ntX = input_transform(tX).numpy()
        else:
            ntX = tX.numpy()

        if outcome_transform is not None:
            ntY = outcome_transform(tY)[0].numpy()
        else:
            ntY = tY.numpy()

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
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        rf.fit(X=ntX, y=ntY.ravel())

        self.model = _RandomForest(rf=rf, output_scaler=outcome_transform)
        if input_transform is not None:
            self.model.input_transform = (  # ty: ignore[invalid-assignment]
                input_transform
            )

    def _dumps(self) -> str:
        """Dumps the random forest to a string via pickle as this is not directly json serializable."""
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str):
        """Loads the actual random forest from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, weights_only=False)
