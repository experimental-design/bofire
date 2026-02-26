
from typing import Any, Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
import botorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate

class TanimotoGPSurrogate(SingleTaskGPSurrogate):

    def __init__(
        self,
        data_model: DataModel,
        pre_computed_tanimoto: bool = True,
        tanimoto_similarity_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.pre_computed_tanimoto = pre_computed_tanimoto
        self.tanimoto_similarity_matrix = tanimoto_similarity_matrix
        super().__init__(data_model=data_model, **kwargs)

    @property
    def re_init_kwargs(self) -> dict:
        return {
            "pre_computed_tanimoto": self.pre_computed_tanimoto,
            "tanimoto_similarity_matrix": self.tanimoto_similarity_matrix,
        }

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
        
        if self.pre_computed_tanimoto:
            if self.tanimoto_similarity_matrix is None:

                covar_module=kernels.map(
                    self.kernel,
                    batch_shape=torch.Size(),
                    active_dims=list(range(n_dim)),
                    features_to_idx_mapper=self.get_feature_indices,
                )
                fingerprints = input_transform.encoders[0].encoding
                self.tanimoto_similarity_matrix = covar_module(fingerprints, fingerprints)


        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim)),
                features_to_idx_mapper=self.get_feature_indices,
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)
