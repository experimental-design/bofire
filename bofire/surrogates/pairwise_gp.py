from typing import Dict, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform

import bofire.kernels.api as kernels
from bofire.data_models.surrogates.api import PairwiseGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.pairwise_trainable import PairwiseTrainableSurrogate
from bofire.utils.torch_tools import tkwargs


class PairwiseGPSurrogate(BotorchSurrogate, PairwiseTrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.engineered_features = data_model.engineered_features
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.PairwiseGP] = None
    training_specs: Dict = {}

    def _fit_pairwise(
        self,
        datapoints: torch.Tensor,
        comparisons: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        **kwargs,
    ):
        if input_transform is not None:
            n_dim = input_transform(datapoints).shape[-1]
        else:
            n_dim = datapoints.shape[-1]

        self.model = botorch.models.PairwiseGP(
            datapoints=datapoints,
            comparisons=comparisons,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=None,
            ),
            input_transform=input_transform,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=self.model.likelihood,
            model=self.model,
        )
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)

    def _predict(self, transformed_X: pd.DataFrame):
        # PairwiseGP silently ignores observation_noise; latent utility has
        # no identifiable observation noise in probit/logit pairwise models.
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            post = self.model.posterior(X=X)
            preds = post.mean.cpu().detach().numpy()
            stds = np.sqrt(post.variance.cpu().detach().numpy())
        return preds, stds
