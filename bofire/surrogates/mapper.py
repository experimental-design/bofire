import warnings
from typing import Callable, Dict, Optional, Type

from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AdditiveMapSaasKernel,
    ICMKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import (
    GaussianLikelihood,
    TaskGaussianLikelihood,
)
from bofire.data_models.means.api import ConstantMean, TaskConstantMean
from bofire.data_models.priors.api import NormalPrior
from bofire.data_models.surrogates import api as data_models
from bofire.surrogates.deterministic import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.fully_bayesian import FullyBayesianSingleTaskGPSurrogate
from bofire.surrogates.map_saas import EnsembleMapSaasSingleTaskGPSurrogate
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.pairwise_gp import PairwiseGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.robust_single_task_gp import RobustSingleTaskGPSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.tanimoto_gp_surrogate import TanimotoGPSurrogate


def map_MixedSingleTaskGPSurrogate(
    data_model: data_models.MixedSingleTaskGPSurrogate,
) -> data_models.SingleTaskGPSurrogate:
    if (
        data_model.continuous_kernel.features is None
        or len(data_model.continuous_kernel.features) == 0
    ):
        # model is purely categorical
        kernel = ScaleKernel(base_kernel=data_model.categorical_kernel)
    else:
        sum_kernel = ScaleKernel(
            base_kernel=AdditiveKernel(
                kernels=[
                    data_model.continuous_kernel,
                    ScaleKernel(base_kernel=data_model.categorical_kernel),
                ]
            )
        )
        product_kernel = ScaleKernel(
            base_kernel=MultiplicativeKernel(
                kernels=[
                    data_model.continuous_kernel,
                    data_model.categorical_kernel,
                ]
            )
        )
        kernel = AdditiveKernel(
            kernels=[
                sum_kernel,
                product_kernel,
            ]
        )
    return data_models.SingleTaskGPSurrogate(
        inputs=data_model.inputs,
        outputs=data_model.outputs,
        categorical_encodings=data_model.categorical_encodings,
        dump=data_model.dump,
        scaler=data_model.scaler,
        output_scaler=data_model.output_scaler,
        likelihood=GaussianLikelihood(
            noise_prior=data_model.noise_prior,
            noise_constraint=data_model.noise_constraint,
        ),
        hyperconfig=None,
        kernel=kernel,
    )


def map_to_SingleTaskGPSurrogate(
    data_model: "data_models.LinearSurrogate | data_models.PolynomialSurrogate",
) -> data_models.SingleTaskGPSurrogate:
    """Express a GP with a fixed kernel and flat noise fields as a single-task GP."""
    return data_models.SingleTaskGPSurrogate(
        inputs=data_model.inputs,
        outputs=data_model.outputs,
        categorical_encodings=data_model.categorical_encodings,
        engineered_features=data_model.engineered_features,
        dump=data_model.dump,
        scaler=data_model.scaler,
        output_scaler=data_model.output_scaler,
        likelihood=GaussianLikelihood(
            noise_prior=data_model.noise_prior,
            noise_constraint=data_model.noise_constraint,
        ),
        hyperconfig=None,
        kernel=data_model.kernel,
    )


def map_AdditiveMapSaasSingleTaskGPSurrogate(
    data_model: data_models.AdditiveMapSaasSingleTaskGPSurrogate,
) -> data_models.SingleTaskGPSurrogate:
    """Express the additive MAP-SAAS GP as a single-task GP built from its components.

    The components are those of BoTorch's ``AdditiveMapSaasSingleTaskGP``: the additive
    SAAS kernel, a constant mean with a standard-normal prior bounded to [-10, 10], and
    the default log-normal noise likelihood.
    """
    return data_models.SingleTaskGPSurrogate(
        inputs=data_model.inputs,
        outputs=data_model.outputs,
        categorical_encodings=data_model.categorical_encodings,
        engineered_features=data_model.engineered_features,
        dump=data_model.dump,
        scaler=data_model.scaler,
        output_scaler=data_model.output_scaler,
        kernel=AdditiveMapSaasKernel(n_taus=data_model.n_taus),
        mean=ConstantMean(prior=NormalPrior(loc=0.0, scale=1.0), bounds=(-10.0, 10.0)),
        likelihood=GaussianLikelihood(),
        hyperconfig=None,
    )


def map_MultiTaskGPSurrogate(
    data_model: data_models.MultiTaskGPSurrogate,
) -> data_models.SingleTaskGPSurrogate:
    """Express the multi-task GP as a single-task GP with task-aware components.

    Its kernel becomes the base of an ICM kernel, and the constant mean and the noise
    are learned per task, as in BoTorch's ``MultiTaskGP``.
    """
    if data_model.task_prior is not None:
        warnings.warn(
            "The LKJ prior has issues when sampling from the prior, prior has been "
            "defaulted to None.",
            UserWarning,
        )
    return data_models.SingleTaskGPSurrogate(
        inputs=data_model.inputs,
        outputs=data_model.outputs,
        categorical_encodings=data_model.categorical_encodings,
        engineered_features=data_model.engineered_features,
        dump=data_model.dump,
        scaler=data_model.scaler,
        output_scaler=data_model.output_scaler,
        kernel=ICMKernel(base_kernel=data_model.kernel),
        mean=TaskConstantMean(),
        likelihood=TaskGaussianLikelihood(
            noise_prior=data_model.noise_prior,
            noise_constraint=data_model.noise_constraint,
        ),
        hyperconfig=None,
    )


DATA_MODEL_MAP: Dict[
    Type[data_models.Surrogate],
    Callable[..., data_models.AnySurrogate],
] = {
    data_models.MixedSingleTaskGPSurrogate: map_MixedSingleTaskGPSurrogate,
    data_models.LinearSurrogate: map_to_SingleTaskGPSurrogate,
    data_models.PolynomialSurrogate: map_to_SingleTaskGPSurrogate,
    data_models.AdditiveMapSaasSingleTaskGPSurrogate: map_AdditiveMapSaasSingleTaskGPSurrogate,
    data_models.MultiTaskGPSurrogate: map_MultiTaskGPSurrogate,
}


SURROGATE_MAP: Dict[Type[data_models.Surrogate], Type[Surrogate]] = {
    data_models.EmpiricalSurrogate: EmpiricalSurrogate,
    data_models.RandomForestSurrogate: RandomForestSurrogate,
    data_models.SingleTaskGPSurrogate: SingleTaskGPSurrogate,
    data_models.RobustSingleTaskGPSurrogate: RobustSingleTaskGPSurrogate,
    data_models.RegressionMLPEnsemble: RegressionMLPEnsemble,
    data_models.ClassificationMLPEnsemble: ClassificationMLPEnsemble,
    data_models.FullyBayesianSingleTaskGPSurrogate: FullyBayesianSingleTaskGPSurrogate,
    data_models.TanimotoGPSurrogate: TanimotoGPSurrogate,
    data_models.LinearDeterministicSurrogate: LinearDeterministicSurrogate,
    data_models.CategoricalDeterministicSurrogate: CategoricalDeterministicSurrogate,
    data_models.EnsembleMapSaasSingleTaskGPSurrogate: EnsembleMapSaasSingleTaskGPSurrogate,
    data_models.PairwiseGPSurrogate: PairwiseGPSurrogate,
}


def register(
    data_model_cls: Type[data_models.Surrogate],
    surrogate_cls: Optional[Type[Surrogate]] = None,
    data_model_transform: Optional[Callable] = None,
):
    """Register a custom surrogate mapping from data model to functional class.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyDataModel)
        class MySurrogate(Surrogate):
            ...

        # Direct call form
        register(MyDataModel, MySurrogate)

    If ``data_model_cls`` is a subclass of
    :class:`~bofire.data_models.surrogates.botorch.BotorchSurrogate`, it is
    also registered with :class:`BotorchSurrogates` so that it can be used
    in botorch-based strategies out of the box.

    Args:
        data_model_cls: The Pydantic data model class.
        surrogate_cls: The functional surrogate class. If not provided,
            returns a decorator.
        data_model_transform: Optional function that transforms the data model
            before instantiation (e.g. to convert to a simpler representation).

    Returns:
        The surrogate class (unchanged) when used as a decorator, None otherwise.
    """

    def _register(cls: Type[Surrogate]) -> Type[Surrogate]:
        # For botorch surrogates, update the data model union first so a
        # discriminator conflict is raised before any map is mutated.
        if issubclass(data_model_cls, data_models.BotorchSurrogate):
            from bofire.data_models.surrogates.botorch_surrogates import (
                register_botorch_surrogate,
            )

            register_botorch_surrogate(data_model_cls)

        SURROGATE_MAP[data_model_cls] = cls
        if data_model_transform is not None:
            DATA_MODEL_MAP[data_model_cls] = data_model_transform

        return cls

    if surrogate_cls is not None:
        _register(surrogate_cls)
        return None

    return _register


def map(data_model: data_models.Surrogate, **kwargs) -> Surrogate:
    new_data_model = data_model
    if data_model.__class__ in DATA_MODEL_MAP:
        new_data_model = DATA_MODEL_MAP[data_model.__class__](data_model)

    cls = SURROGATE_MAP[new_data_model.__class__]
    return cls(data_model=new_data_model, **kwargs)
