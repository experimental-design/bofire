from typing import Any, List, Optional

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor

from bofire.utils.torch_tools import tkwargs


class RF(Model):
    def __init__(self, rf: RandomForestRegressor):
        super().__init__()
        self.rf = rf

    def forward(self, X: Tensor):
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
            for estimator in self.rf.estimators_:
                batch_preds.append(
                    estimator.predict(nX[i]).reshape((nX[i].shape[0], 1))
                )
            preds.append(np.stack(batch_preds, axis=-1))
        preds = np.stack(preds, axis=0)
        if X.ndim == 3:
            return torch.from_numpy(preds).to(**tkwargs)
        else:
            return torch.from_numpy(preds).to(**tkwargs).squeeze(dim=0)

        # tpreds = torch.from_numpy(preds).to(**tkwargs)
        # if X
        # for estimator in self.rf.estimators_:
        #    preds.append(estimator.predict(nX).reshape((nX.shape[0], 1)))
        # preds = np.stack(preds, axis=-1)
        # return torch.from_numpy(preds).to(**tkwargs)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> EnsemblePosterior:
        r"""Compute the (deterministic) posterior at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior. If omitted, computes the posterior
                over all model outputs.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `DeterministicPosterior` object, representing `batch_shape` joint
            posteriors over `n` points and the outputs selected by `output_indices`.
        """
        # Apply the input transforms in `eval` mode.
        self.eval()
        X = self.transform_inputs(X)
        values = self.forward(X)
        # NOTE: The `outcome_transform` `untransform`s the predictions rather than the
        # `posterior` (as is done in GP models). This is more general since it works
        # even if the transform doesn't support `untransform_posterior`.
        if hasattr(self, "outcome_transform"):
            values, _ = self.outcome_transform.untransform(values)
        if output_indices is not None:
            values = values[..., output_indices]
        posterior = EnsemblePosterior(values=values)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        else:
            return posterior
