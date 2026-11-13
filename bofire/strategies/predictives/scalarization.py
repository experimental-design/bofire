"""Mapping of scalarization data models to BoTorch objective callables."""

from typing import Callable, Dict, Optional, Type

import pandas as pd
import torch
from torch import Tensor

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import (
    get_additive_botorch_objective,
    get_multiplicative_additive_objective,
    get_multiplicative_botorch_objective,
    get_objective_callable,
    tkwargs,
)


ObjectiveCallable = Callable[[Tensor, Tensor], Tensor]

SCALARIZATION_MAP: Dict[Type[data_models.Scalarization], Callable] = {}


def register(
    data_model_cls: Type[data_models.Scalarization],
    map_fn: Optional[Callable] = None,
):
    """Register a custom scalarization mapping from data model to objective callable.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyScalarization)
        def map_my_scalarization(
            data_model, domain, experiments, input_preprocessing_specs
        ):
            return lambda samples, X: ...

        # Direct call form
        register(MyScalarization, map_my_scalarization)

    The returned callable receives the sampled output values (`samples`, outputs
    in the order of `domain.outputs`) and the corresponding inputs `X`, transformed
    according to `input_preprocessing_specs`, and returns one value per sample to
    maximize.

    Args:
        data_model_cls: The scalarization data model class.
        map_fn: A callable that takes ``(data_model, domain, experiments,
            input_preprocessing_specs)`` and returns the objective callable. If not
            provided, returns a decorator.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.
    """

    def _register(fn: Callable) -> Callable:
        # Register with the data model union first so a discriminator conflict
        # is raised before the functional map is touched (no partial state).
        data_models.register_scalarization(data_model_cls)
        SCALARIZATION_MAP[data_model_cls] = fn
        return fn

    if map_fn is not None:
        _register(map_fn)
        return None
    return _register


def map(
    data_model: data_models.Scalarization,
    domain: Domain,
    experiments: pd.DataFrame,
    input_preprocessing_specs: InputTransformSpecs,
) -> ObjectiveCallable:
    """Map a scalarization data model to the objective callable for BoTorch.

    Args:
        data_model: The scalarization.
        domain: The domain of the strategy.
        experiments: The experiments, used to adapt objectives to the data.
        input_preprocessing_specs: How the inputs are transformed for the model.

    Returns:
        A callable mapping sampled outputs and inputs to the value to maximize.
    """
    return SCALARIZATION_MAP[data_model.__class__](
        data_model, domain, experiments, input_preprocessing_specs
    )


@register(data_models.SingleObjectiveScalarization)
def map_SingleObjectiveScalarization(
    data_model: data_models.SingleObjectiveScalarization,
    domain: Domain,
    experiments: pd.DataFrame,
    input_preprocessing_specs: InputTransformSpecs,
) -> ObjectiveCallable:
    try:
        target_feature = domain.outputs.get_by_objective(
            excludes=ConstrainedObjective,
        )[0]
    except IndexError:
        target_feature = domain.outputs.get_by_objective(includes=Objective)[0]
    target_index = domain.outputs.get_keys().index(target_feature.key)
    x_adapt = torch.from_numpy(
        domain.outputs.preprocess_experiments_one_valid_output(
            target_feature.key,
            experiments,
        )[target_feature.key].values,
    ).to(**tkwargs)
    return get_objective_callable(
        idx=target_index,
        objective=target_feature.objective,
        x_adapt=x_adapt,
    )


@register(data_models.WeightedSumScalarization)
def map_WeightedSumScalarization(
    data_model: data_models.WeightedSumScalarization,
    domain: Domain,
    experiments: pd.DataFrame,
    input_preprocessing_specs: InputTransformSpecs,
) -> ObjectiveCallable:
    return get_additive_botorch_objective(
        outputs=domain.outputs,
        experiments=experiments,
        exclude_constraints=data_model.separates_output_constraints(),
        weights=data_model.get_weights(domain),
    )


@register(data_models.WeightedProductScalarization)
def map_WeightedProductScalarization(
    data_model: data_models.WeightedProductScalarization,
    domain: Domain,
    experiments: pd.DataFrame,
    input_preprocessing_specs: InputTransformSpecs,
) -> ObjectiveCallable:
    return get_multiplicative_botorch_objective(
        outputs=domain.outputs,
        experiments=experiments,
        weights=data_model.get_weights(domain),
        weight_normalization=data_model.weight_normalization,
    )


@register(data_models.WeightedProductSumScalarization)
def map_WeightedProductSumScalarization(
    data_model: data_models.WeightedProductSumScalarization,
    domain: Domain,
    experiments: pd.DataFrame,
    input_preprocessing_specs: InputTransformSpecs,
) -> ObjectiveCallable:
    return get_multiplicative_additive_objective(
        outputs=domain.outputs,
        experiments=experiments,
        exclude_constraints=False,
        additive_features=data_model.additive_features,
        weights=data_model.get_weights(domain),
        weight_normalization=data_model.weight_normalization,
    )
