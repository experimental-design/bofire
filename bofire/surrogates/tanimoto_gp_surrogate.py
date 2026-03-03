from typing import Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate


class TanimotoGPSurrogate(SingleTaskGPSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        tanimoto_similarity_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.tanimoto_calculation_mode = data_model.tanimoto_calculation_mode
        self.tanimoto_similarity_matrix = tanimoto_similarity_matrix
        super().__init__(data_model=data_model, **kwargs)

    @property
    def re_init_kwargs(self) -> dict:
        re_init_kwargs = super().re_init_kwargs
        re_init_kwargs.update(
            {
                "tanimoto_similarity_matrix": self.tanimoto_similarity_matrix,
            }
        )
        return re_init_kwargs

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

        if self.tanimoto_calculation_mode == "pre_computed":
            if self.tanimoto_similarity_matrix is None:
                covar_module_pre_calc = kernels.map(
                    self.kernel,
                    batch_shape=torch.Size(),
                    active_dims=list(range(n_dim)),
                    features_to_idx_mapper=self.get_feature_indices,
                    pre_computed_tanimoto=False,
                )  # temporary kernel to compute the tanimoto similarity matrix
                fingerprints = input_transform.encoders[0].encoding
                cov_module = covar_module_pre_calc
                while not hasattr(cov_module, "pre_computed_tanimoto"):
                    cov_module = cov_module.base_kernel
                cov = cov_module.forward(fingerprints, fingerprints)
                self.tanimoto_similarity_matrix = cov.detach()

            input_transform_use = (
                None  # We will only pass index-based inputs to the model
            )
        else:
            input_transform_use = input_transform

        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(n_dim))
                if not (self.tanimoto_calculation_mode == "pre_computed")
                else [0],  # active dims are not needed if using pre-computed tanimoto
                features_to_idx_mapper=self.get_feature_indices,
                pre_computed_tanimoto=(
                    self.tanimoto_calculation_mode == "pre_computed"
                ),
                tanimoto_similarity_matrix=self.tanimoto_similarity_matrix,
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform_use,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)
