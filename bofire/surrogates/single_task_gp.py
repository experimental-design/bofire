from typing import Dict, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate


class SingleTaskGPSurrogate(TrainableBotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.noise_prior = data_model.noise_prior
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        if input_transform is not None:
            n_dim = input_transform(tX).shape[-1]
        else:
            n_dim = tX.shape[-1]

        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.categorical_encodings, feats
                ),
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)
