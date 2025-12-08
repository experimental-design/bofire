from typing import Dict, List, Optional, cast

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing_extensions import Self

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.kernels.api import AnyKernel
from bofire.data_models.priors.api import AnyPrior
from bofire.data_models.surrogates.api import ScalerEnum

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPHyperconfig
from bofire.data_models.surrogates.trainable import AnyAggregation
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.model_utils import make_surrogate


class SingleTaskGPSurrogate(TrainableBotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.noise_prior = data_model.noise_prior
        self.hyperconfig = data_model.hyperconfig
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

    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        hyperconfig: Optional[SingleTaskGPHyperconfig] = None,
        aggregations: Optional[List[AnyAggregation]] = None,
        input_preprocessing_specs: Optional[InputTransformSpecs] = None,
        dump: Optional[str] = None,
        categorical_encodings: Optional[InputTransformSpecs] = None,
        scaler: ScalerEnum = ScalerEnum.NORMALIZE,
        output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE,
        kernel: Optional[AnyKernel] = None,
        noise_prior: Optional[AnyPrior] = None,
    ) -> Self:
        """
        Factory method to create a SingleTaskGPSurrogate from a data model.
        Args:
            hyperconfig: SingleTaskGPHyperconfig or None
            aggregations: List[AnyAggregation] or None
            inputs: Inputs
            outputs: Outputs
            input_preprocessing_specs: InputTransformSpecs
            dump: str or None
            categorical_encodings: InputTransformSpecs
            scaler: ScalerEnum
            output_scaler: ScalerEnum
            kernel: AnyKernel
            noise_prior: AnyPrior
        Returns:
            SingleTaskGPSurrogate: A new instance.
        """
        return cast(Self, make_surrogate(cls, DataModel, locals()))
