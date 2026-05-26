from typing import Dict, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform

import bofire.kernels.api as kernels
from bofire.data_models.surrogates.api import PairwiseGPSurrogate as DataModel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.pairwise_trainable import PairwiseTrainableSurrogate


# maps the serializable likelihood name to the BoTorch PairwiseLikelihood class
PAIRWISE_LIKELIHOODS = {
    "probit": PairwiseProbitLikelihood,
    "logit": PairwiseLogitLikelihood,
}


class PairwiseGPSurrogate(BotorchSurrogate, PairwiseTrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.likelihood = data_model.likelihood
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
            likelihood=PAIRWISE_LIKELIHOODS[self.likelihood](),
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=self.get_feature_indices,
            ),
            input_transform=input_transform,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=self.model.likelihood,
            model=self.model,
        )
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)

    # `_predict` is inherited from BotorchSurrogate: it calls
    # `posterior(X, observation_noise=True)`, and PairwiseGP ignores
    # `observation_noise` (verified in scripts/pairwise_gp_checks.py).
