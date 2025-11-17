from typing import Dict, Type

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
from bofire.surrogates.map_saas import AdditiveMapSaasSingleTaskGPSurrogate
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.robust_single_task_gp import RobustSingleTaskGPSurrogate
from bofire.surrogates.shape import PiecewiseLinearGPSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate


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
                    data_model.continuous_kernel,  # type: ignore
                    ScaleKernel(base_kernel=data_model.categorical_kernel),
                ]
            )
        )
        product_kernel = ScaleKernel(
            base_kernel=MultiplicativeKernel(
                kernels=[
                    data_model.continuous_kernel,  # type: ignore
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
        hyperconfig=None,
        kernel=kernel,
    )


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
    data_models.TanimotoGPSurrogate: SingleTaskGPSurrogate,
    data_models.LinearDeterministicSurrogate: LinearDeterministicSurrogate,
    data_models.MultiTaskGPSurrogate: MultiTaskGPSurrogate,
    data_models.SingleTaskIBNNSurrogate: SingleTaskGPSurrogate,
    data_models.PiecewiseLinearGPSurrogate: PiecewiseLinearGPSurrogate,
    data_models.CategoricalDeterministicSurrogate: CategoricalDeterministicSurrogate,
    data_models.AdditiveMapSaasSingleTaskGPSurrogate: AdditiveMapSaasSingleTaskGPSurrogate,
}


def map(data_model: data_models.Surrogate) -> Surrogate:
    new_data_model = data_model
    if isinstance(data_model, data_models.MixedSingleTaskGPSurrogate):
        new_data_model = map_MixedSingleTaskGPSurrogate(data_model)
    cls = SURROGATE_MAP[new_data_model.__class__]
    return cls(data_model=new_data_model)
