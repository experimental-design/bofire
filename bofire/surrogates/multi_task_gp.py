from typing import Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.means import MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.likelihoods.api as likelihoods
import bofire.means.api as means
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.features.api import CategoricalTaskInput
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.utils import scipy_options
from bofire.utils.torch_tools import tkwargs


class MultiTaskGPSurrogate(TrainableBotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_tasks = len(
            data_model.inputs.get(CategoricalTaskInput).features[0].categories
        )
        self.kernel = data_model.kernel
        self.optimizer = data_model.optimizer
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.mean = data_model.mean
        self.likelihood = data_model.likelihood
        # obtain the name of the task feature
        self.task_feature_key = data_model.inputs.get_keys(CategoricalTaskInput)[0]

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MultiTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ) -> None:
        n_dim = tX.shape[-1]
        task_feature = self.get_feature_indices([self.task_feature_key])[0]

        self.model = botorch.models.MultiTaskGP(
            train_X=tX,
            train_Y=tY,
            task_feature=task_feature,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=self.get_feature_indices(
                    [
                        key
                        for key in self.inputs.get_keys()
                        if key != self.task_feature_key
                    ]
                    + self.engineered_features.get_keys(),
                ),
                features_to_idx_mapper=self.get_feature_indices,
            ),
            mean_module=MultitaskMean(
                means.map(self.mean, d=n_dim), num_tasks=self.n_tasks
            ),
            likelihood=likelihoods.map(
                self.likelihood,
                d=n_dim,
                num_tasks=self.n_tasks,
                task_feature_index=task_feature,
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            # Pass None explicitly to avoid the default BetaPrior introduced in
            # botorch PR #3271, which registers lambdas in PositiveIndexKernel
            # (PR #3267) and breaks torch.save-based serialization.
            task_covar_prior=None,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": scipy_options(self.optimizer)},
            max_attempts=self.optimizer.max_attempts,
        )

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)
            except NotImplementedError:
                posterior = self.model.posterior(X=X, observation_noise=False)
            preds = posterior.mean.cpu().detach().numpy()
            stds = np.sqrt(posterior.variance.cpu().detach().numpy())

        return preds, stds
