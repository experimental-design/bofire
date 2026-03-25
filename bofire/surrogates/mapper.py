from typing import Callable, Dict, Optional, Type

from bofire.data_models.kernels.api import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.surrogates import api as data_models
from bofire.surrogates.deterministic import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.fully_bayesian import FullyBayesianSingleTaskGPSurrogate
from bofire.surrogates.map_saas import (
    AdditiveMapSaasSingleTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
)
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.robust_single_task_gp import RobustSingleTaskGPSurrogate
from bofire.surrogates.shape import PiecewiseLinearGPSurrogate
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
        input_preprocessing_specs=data_model.input_preprocessing_specs,
        categorical_encodings=data_model.categorical_encodings,
        dump=data_model.dump,
        scaler=data_model.scaler,
        output_scaler=data_model.output_scaler,
        noise_prior=data_model.noise_prior,
        noise_constraint=data_model.noise_constraint,
        hyperconfig=None,
        kernel=kernel,
    )


DATA_MODEL_MAP: Dict[
    Type[data_models.MixedSingleTaskGPSurrogate],
    Callable[[data_models.MixedSingleTaskGPSurrogate], data_models.AnySurrogate],
] = {
    data_models.MixedSingleTaskGPSurrogate: map_MixedSingleTaskGPSurrogate,
}


SURROGATE_MAP: Dict[Type[data_models.Surrogate], Type[Surrogate]] = {
    data_models.EmpiricalSurrogate: EmpiricalSurrogate,
    data_models.RandomForestSurrogate: RandomForestSurrogate,
    data_models.SingleTaskGPSurrogate: SingleTaskGPSurrogate,
    data_models.RobustSingleTaskGPSurrogate: RobustSingleTaskGPSurrogate,
    data_models.RegressionMLPEnsemble: RegressionMLPEnsemble,
    data_models.ClassificationMLPEnsemble: ClassificationMLPEnsemble,
    data_models.FullyBayesianSingleTaskGPSurrogate: FullyBayesianSingleTaskGPSurrogate,
    data_models.LinearSurrogate: SingleTaskGPSurrogate,
    data_models.PolynomialSurrogate: SingleTaskGPSurrogate,
    data_models.TanimotoGPSurrogate: TanimotoGPSurrogate,
    data_models.LinearDeterministicSurrogate: LinearDeterministicSurrogate,
    data_models.MultiTaskGPSurrogate: MultiTaskGPSurrogate,
    data_models.SingleTaskIBNNSurrogate: SingleTaskGPSurrogate,
    data_models.PiecewiseLinearGPSurrogate: PiecewiseLinearGPSurrogate,
    data_models.CategoricalDeterministicSurrogate: CategoricalDeterministicSurrogate,
    data_models.AdditiveMapSaasSingleTaskGPSurrogate: AdditiveMapSaasSingleTaskGPSurrogate,
    data_models.EnsembleMapSaasSingleTaskGPSurrogate: EnsembleMapSaasSingleTaskGPSurrogate,
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
        SURROGATE_MAP[data_model_cls] = cls
        if data_model_transform is not None:
            DATA_MODEL_MAP[data_model_cls] = data_model_transform

        if issubclass(data_model_cls, data_models.BotorchSurrogate):
            from bofire.data_models.surrogates.botorch_surrogates import (
                register_botorch_surrogate,
            )

            register_botorch_surrogate(data_model_cls)
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
