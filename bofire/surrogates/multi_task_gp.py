from typing import Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing_extensions import Self

import bofire.kernels.api as kernels
import bofire.likelihoods.api as likelihoods
import bofire.means.api as means
from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.features.api import CategoricalTaskInput
from bofire.data_models.kernels.api import AnyKernel, RBFKernel
from bofire.data_models.likelihoods.api import (
    HVARFNER_LIKELIHOOD,
    AnyMultiTaskLikelihood,
    TaskwiseLikelihood,
)
from bofire.data_models.means.api import AnyMultiTaskMean, ConstantMean, TaskwiseMean
from bofire.data_models.priors.api import HVARFNER_LENGTHSCALE_PRIOR, GreaterThan
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import make_surrogate
from bofire.surrogates.utils import scipy_options
from bofire.utils.torch_tools import tkwargs


class MultiTaskGPSurrogate(TrainableBotorchSurrogate):
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
        kernel: AnyKernel = RBFKernel(  # noqa: B008 - copied by make_surrogate
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),  # noqa: B008 - copied by make_surrogate
            lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),  # noqa: B008 - copied by make_surrogate
        ),
        mean: AnyMultiTaskMean = TaskwiseMean(mean=ConstantMean()),  # noqa: B008 - copied by make_surrogate
        likelihood: AnyMultiTaskLikelihood = TaskwiseLikelihood(  # noqa: B008 - copied by make_surrogate
            likelihood=HVARFNER_LIKELIHOOD()  # noqa: B008 - copied by make_surrogate
        ),
        optimizer: LBFGSB = LBFGSB(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create a runtime multi-task GP using validated data-model settings.

        Args:
            inputs: Input features, including exactly one categorical task input.
            outputs: Output features.
            dump: Serialized fitted model to restore, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.
            scaler: Input scaler; None disables scaling.
            output_scaler: Output scaling method.
            kernel: Covariance kernel.
            mean: Taskwise mean function, or a plain mean shared by all tasks.
            likelihood: Taskwise noise model, or a Gaussian likelihood shared by all tasks.
            optimizer: Settings for fitting the GP hyperparameters.

        Defaults mirror the data model and are copied for each call. Call ``fit`` to train.
        """
        return make_surrogate(cls, DataModel, locals())

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
        if input_transform is not None:
            n_dim = input_transform(tX).shape[-1]
        else:
            n_dim = tX.shape[-1]
        d = n_dim - 1  # the inputs the model sees, without the task column
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
                inputs=self.inputs,
            ),
            mean_module=means.map(self.mean, d=d, num_tasks=self.n_tasks),
            likelihood=likelihoods.map(
                self.likelihood,
                d=d,
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
