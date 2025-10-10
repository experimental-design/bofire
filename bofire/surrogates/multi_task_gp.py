import warnings
from typing import Dict, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.features.api import TaskInput
from bofire.data_models.priors.api import LKJPrior

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.utils.torch_tools import tkwargs


class MultiTaskGPSurrogate(TrainableBotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_tasks = len(data_model.inputs.get(TaskInput).features[0].categories)
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.noise_prior = data_model.noise_prior
        self.task_prior = data_model.task_prior
        if isinstance(self.task_prior, LKJPrior):
            # set the number of tasks in the prior
            self.task_prior.n_tasks = self.n_tasks
        # obtain the name of the task feature
        self.task_feature_key = data_model.inputs.get_keys(TaskInput)[0]

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MultiTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ) -> None:  # type: ignore
        if input_transform is not None:
            n_dim = input_transform(tX).shape[-1]
        else:
            n_dim = tX.shape[-1]

        self.model = botorch.models.MultiTaskGP(
            train_X=tX,
            train_Y=tY,
            task_feature=self.inputs.get_feature_indices(
                self.categorical_encodings, [self.task_feature_key]
            )[0],
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(
                    range(n_dim - 1),
                ),  # kernel is for input space so we subtract one for the fidelity index
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.categorical_encodings, feats
                ),
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        if isinstance(self.task_prior, LKJPrior):
            warnings.warn(
                "The LKJ prior has issues when sampling from the prior, prior has been defaulted to None.",
                UserWarning,
            )
            # once the issue is fixed, the following line should be uncommented
            # self.model.task_covar_module.register_prior(
            #     "IndexKernelPrior", priors.map(self.lkj_prior), _index_kernel_prior_closure
            # )
        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore
            except NotImplementedError:
                posterior = self.model.posterior(X=X, observation_noise=False)  # type: ignore
            preds = posterior.mean.cpu().detach().numpy()
            stds = np.sqrt(posterior.variance.cpu().detach().numpy())

        return preds, stds


def _index_kernel_prior_closure(m):
    return m._eval_covar_matrix()
