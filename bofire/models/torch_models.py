import itertools
from abc import abstractmethod
from typing import Dict, List, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelList
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model as BotorchBaseModel
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputStandardize,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import Kernel as GpytorchKernel
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from pydantic import validator

from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    InputFeatures,
    NumericalInput,
    OutputFeatures,
    TInputTransformSpecs,
)
from bofire.domain.util import PydanticBaseModel
from bofire.models.model import Model, TrainableModel
from bofire.utils.enum import CategoricalEncodingEnum, OutputFilteringEnum, ScalerEnum
from bofire.utils.torch_tools import OneHotToNumeric, tkwargs


def get_dim_subsets(d: int, active_dims: List[int], cat_dims: List[int]):
    def check_indices(d, indices):
        if len(set(indices)) != len(indices):
            raise ValueError("Elements of `indices` list must be unique!")
        if any([i > d - 1 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        if len(indices) > d:
            raise ValueError("Can provide at most `d` indices!")
        if any([i < 0 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        return indices

    if len(active_dims) == 0:
        raise ValueError("At least one active dim has to be provided!")

    # check validity of indices
    active_dims = check_indices(d, active_dims)
    cat_dims = check_indices(d, cat_dims)

    # compute subsets
    ord_dims = sorted(set(range(d)) - set(cat_dims))
    ord_active_dims = sorted(
        set(active_dims) - set(cat_dims)
    )  # includes also descriptors
    cat_active_dims = sorted([i for i in cat_dims if i in active_dims])
    return ord_dims, ord_active_dims, cat_active_dims


class BaseKernel(PydanticBaseModel):
    @abstractmethod
    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        pass


class ContinuousKernel(BaseKernel):
    pass


class CategoricalKernel(BaseKernel):
    pass


class HammondDistanceKernel(CategoricalKernel):
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        raise NotImplementedError


class RBF(ContinuousKernel):
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        return RBFKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(active_dims) if self.ard else None,
            active_dims=active_dims,  # type: ignore
        )


class Matern(ContinuousKernel):
    ard: bool = True
    nu: float = 2.5

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        return MaternKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(active_dims) if self.ard else None,
            active_dims=active_dims,
            nu=self.nu,
        )


class Linear(ContinuousKernel):
    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        return LinearKernel(batch_shape=batch_shape)


class BotorchModel(Model):

    model: Optional[BotorchBaseModel]

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        input_features = values["input_features"]
        categorical_keys = input_features.get_keys(CategoricalInput, exact=True)
        descriptor_keys = input_features.get_keys(
            CategoricalDescriptorInput, exact=True
        )
        for key in categorical_keys:
            if (
                v.get(key, CategoricalEncodingEnum.ONE_HOT)
                != CategoricalEncodingEnum.ONE_HOT
            ):
                raise ValueError(
                    "Botorch based models have to use one hot encodings for categoricals"
                )
            else:
                v[key] = CategoricalEncodingEnum.ONE_HOT
        # TODO: include descriptors into probabilistic reparam via OneHotToDescriptor input transform
        for key in descriptor_keys:
            if v.get(key, CategoricalEncodingEnum.DESCRIPTOR) not in [
                CategoricalEncodingEnum.DESCRIPTOR,
                CategoricalEncodingEnum.ONE_HOT,
            ]:
                raise ValueError(
                    "Botorch based models have to use one hot encodings or descriptor encodings for categoricals."
                )
            elif v.get(key) is None:
                v[key] = CategoricalEncodingEnum.DESCRIPTOR
        for key in input_features.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError(
                    "Botorch based models have to use internal transforms to preprocess numerical features."
                )
        return v

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            preds = self.model.posterior(X=X, observation_noise=True).mean.cpu().detach().numpy()  # type: ignore
            stds = np.sqrt(self.model.posterior(X=X, observation_noise=True).variance.cpu().detach().numpy())  # type: ignore
        return preds, stds


class BotorchModels(PydanticBaseModel):

    models: List[BotorchModel]

    @validator("models")
    def validate_models(cls, v, values):
        # validate that all models are single output models
        # TODO: this restriction has to be removed at some point
        for model in v:
            if len(model.output_features) != 1:
                raise ValueError("Only single output models allowed.")
        # check that the output feature keys are distinct
        used_output_feature_keys = list(
            itertools.chain.from_iterable(
                [model.output_features.get_keys() for model in v]
            )
        )
        if len(set(used_output_feature_keys)) != len(used_output_feature_keys):
            raise ValueError("Output feature keys are not unique across models.")
        # get the feature keys present in all models
        used_feature_keys = []
        for model in v:
            for key in model.input_features.get_keys():
                if key not in used_feature_keys:
                    used_feature_keys.append(key)
        # check that the features and preprocessing steps are equal trough the models
        for key in used_feature_keys:
            features = [
                model.input_features.get_by_key(key)
                for model in v
                if key in model.input_features.get_keys()
            ]
            preproccessing = [
                model.input_preprocessing_specs[key]
                for model in v
                if key in model.input_preprocessing_specs
            ]
            if all(features) is False:
                raise ValueError(f"Features with key {key} are incompatible.")
            if len(set(preproccessing)) > 1:
                raise ValueError(
                    f"Preprocessing steps for features with {key} are incompatible."
                )
        return v

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        return {
            key: value
            for model in self.models
            for key, value in model.input_preprocessing_specs.items()
        }

    def fit(self, experiments: pd.DataFrame):
        for model in self.models:
            if isinstance(model, TrainableModel):
                model.fit(experiments)

    @property
    def output_features(self) -> OutputFeatures:
        return OutputFeatures(
            features=list(
                itertools.chain.from_iterable(
                    [model.output_features.get() for model in self.models]  # type: ignore
                )
            )
        )

    def _check_compability(
        self, input_features: InputFeatures, output_features: OutputFeatures
    ):
        # TODO: add sync option
        used_output_feature_keys = self.output_features.get_keys()
        if sorted(used_output_feature_keys) != sorted(output_features.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.models):
            if len(model.input_features) > len(input_features):
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
                )
            for feat in model.input_features:
                try:
                    other_feat = input_features.get_by_key(feat.key)
                except KeyError:
                    raise ValueError(f"Feature {feat.key} not found.")
                # now compare the features
                # TODO: make more sohisticated comparisons based on the type
                # has to to be implemented in features, for the start
                # we go with __eq__
                if feat != other_feat:
                    raise ValueError(f"Features with key {feat.key} are incompatible.")
                if feat.key not in used_feature_keys:
                    used_feature_keys.append(feat.key)
        if len(used_feature_keys) != len(input_features):
            raise ValueError("Unused features are present.")

    def compatibilize(
        self, input_features: InputFeatures, output_features: OutputFeatures
    ) -> ModelList:
        # TODO: add sync option
        # check if models are compatible to provided inputs and outputs
        # of the optimization domain
        self._check_compability(
            input_features=input_features, output_features=output_features
        )
        features2idx, _ = input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        #
        all_gp = True
        botorch_models = []
        # we sort the models by sorting them with their occurence in output_features
        for output_feature_key in output_features.get_keys():
            # get the corresponding model
            model = {model.output_features[0].key: model for model in self.models}[
                output_feature_key
            ]
            # in case that inputs are complete we do not need to adjust anything
            if len(model.input_features) == len(input_features):
                botorch_models.append(model.model)
            # in this case we have to care for the indices
            if len(model.input_features) < len(input_features):
                indices = []
                for key in model.input_features.get_keys():
                    indices += features2idx[key]
                features_filter = FilterFeatures(
                    feature_indices=torch.tensor(indices, dtype=torch.int64),
                    transform_on_train=False,
                )
                if (
                    hasattr(model.model, "input_transform")
                    and model.model.input_transform is not None  # type: ignore
                ):
                    model.model.input_transform = ChainedInputTransform(  # type: ignore
                        tf1=features_filter, tf2=model.model.input_transform  # type: ignore
                    )
                else:
                    model.model.input_transform = features_filter  # type: ignore

                botorch_models.append(model.model)
                if isinstance(model.model, botorch.models.SingleTaskGP) is False:
                    all_gp = False

        if len(botorch_models) == 1:
            return botorch_models[0]
        if all_gp:
            return botorch.models.ModelListGP(*botorch_models)
        return ModelList(*botorch_models)


class SingleTaskGPModel(BotorchModel, TrainableModel):
    kernel: ContinuousKernel = Matern(ard=True, nu=2.5)
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = (
        OutputFilteringEnum.ALL
    )  # only relevant for training
    # features2idx: Optional[Dict] = None  # only relevant for training
    # non_numerical_features: Optional[List] = None  # only relevant for training
    training_specs: Dict = {}  # only relevant for training

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]
        # transform X
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        # transform from pandas to torch
        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        if tX.dim() == 2:
            batch_shape = torch.Size()
        else:
            batch_shape = torch.Size([tX.shape[0]])

        d = tX.shape[-1]

        cat_dims = []
        for feat in non_numerical_features:
            cat_dims += features2idx[feat]

        ord_dims, _, _ = get_dim_subsets(
            d=d, active_dims=list(range(d)), cat_dims=cat_dims
        )
        # first get the scaler
        # TODO use here the correct bounds
        if self.scaler == ScalerEnum.NORMALIZE:
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )

            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                batch_shape=batch_shape,
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims if len(ord_dims) != d else None,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError("Scaler enum not known.")

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=ScaleKernel(
                self.kernel.to_gpytorch(
                    batch_shape=batch_shape,
                    active_dims=list(range(d)),
                    ard_num_dims=1,
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            ),
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=scaler,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)


class MixedSingleTaskGPModel(BotorchModel, TrainableModel):
    continuous_kernel: ContinuousKernel = Matern(ard=True, nu=2.5)
    categorical_kernel: CategoricalKernel = HammondDistanceKernel(ard=True)
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    model: Optional[botorch.models.MixedSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    # features2idx: Optional[Dict] = None
    # non_numerical_features: Optional[List] = None
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # get transform meta information
        features2idx, _ = self.input_features._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
        ]
        # transform X
        transformed_X = self.input_features.transform(X, self.input_preprocessing_specs)

        # transform from pandas to torch
        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        if tX.dim() == 2:
            batch_shape = torch.Size()
        else:
            batch_shape = torch.Size([tX.shape[0]])

        # get indices of the continuous and categorical dims
        d = tX.shape[-1]

        ord_dims = []
        for feat in self.input_features.get():
            if feat.key not in non_numerical_features:
                ord_dims += features2idx[feat.key]
        cat_dims = list(
            range(len(ord_dims), len(ord_dims) + len(non_numerical_features))
        )
        categorical_features = {
            features2idx[feat][0]: len(features2idx[feat])
            for feat in non_numerical_features
        }

        # first get the scaler
        if self.scaler == ScalerEnum.NORMALIZE:
            # TODO: take the real bounds here
            lower, upper = self.input_features.get_bounds(
                specs=self.input_preprocessing_specs, experiments=X
            )
            scaler = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                indices=ord_dims,
                batch_shape=batch_shape,
            )
        elif self.scaler == ScalerEnum.STANDARDIZE:
            scaler = InputStandardize(
                d=d,
                indices=ord_dims,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError("Scaler enum not known.")

        o2n = OneHotToNumeric(
            dim=d, categorical_features=categorical_features, transform_on_train=False
        )
        tf = ChainedInputTransform(tf1=scaler, tf2=o2n)

        # fit the model
        self.model = botorch.models.MixedSingleTaskGP(
            train_X=o2n.transform(tX),
            train_Y=tY,
            cat_dims=cat_dims,
            cont_kernel_factory=self.continuous_kernel.to_gpytorch,
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=tf,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)


class EmpiricalModel(BotorchModel):
    """All necessary functions has to be implemented in the model which can then be loaded
    from cloud pickle.

    Attributes:
        model (DeterministicModel): Botorch model instance.
    """

    model: Optional[DeterministicModel] = None
