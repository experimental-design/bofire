import inspect
from typing import Literal, Type
from unittest.mock import MagicMock

import pandas as pd

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    EngineeredFeature,
    Feature,
)
from bofire.data_models.kernels.continuous import ContinuousKernel as _ContinuousBase
from bofire.data_models.kernels.kernel import Kernel as KernelDataModel
from bofire.data_models.priors.prior import Prior as PriorDataModel
from bofire.data_models.strategies.strategy import Strategy as StrategyDataModel
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.surrogate import Surrogate as SurrogateDataModel
from bofire.strategies.strategy import Strategy
from bofire.surrogates.surrogate import Surrogate


# ---------------------------------------------------------------------------
# Stub data model and functional classes for strategies
# ---------------------------------------------------------------------------


class _CustomStrategyDataModel(StrategyDataModel):
    type: str = "CustomStrategy"

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True


class _CustomStrategy(Strategy):
    def _ask(self, candidate_count):
        return pd.DataFrame()

    def has_sufficient_experiments(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Stub data model and functional classes for surrogates
# ---------------------------------------------------------------------------

_INPUTS = Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))])
_OUTPUTS = Outputs(features=[ContinuousOutput(key="y")])


class _CustomSurrogateDataModel(SurrogateDataModel):
    type: str = "CustomSurrogate"
    inputs: Inputs = _INPUTS
    outputs: Outputs = _OUTPUTS

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        return True


class _CustomSurrogate(Surrogate):
    def __init__(self, data_model, **kwargs):
        # simplified init that skips heavy base-class logic
        self.data_model = data_model

    def predict(self, X):
        pass

    def _predict(self, transformed_X):
        pass

    def loads(self, data):
        pass

    def dumps(self):
        return ""

    def _dumps(self):
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_domain():
    return Domain(
        inputs=Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))]),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterStrategy:
    def test_register_and_map(self):
        import bofire.strategies.api as strategies_api
        from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP

        # ensure not registered yet
        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)

        strategies_api.register(_CustomStrategyDataModel, _CustomStrategy)
        assert ACTUAL_MAP[_CustomStrategyDataModel] is _CustomStrategy

        # round-trip through strategies.map
        dm = _CustomStrategyDataModel(domain=_make_domain())
        result = strategies_api.map(dm)
        assert isinstance(result, _CustomStrategy)

        # cleanup
        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)

    def test_register_decorator_syntax(self):
        import bofire.strategies.api as strategies_api
        from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP

        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)

        @strategies_api.register(_CustomStrategyDataModel)
        class _DecoratedStrategy(_CustomStrategy):
            pass

        assert ACTUAL_MAP[_CustomStrategyDataModel] is _DecoratedStrategy

        dm = _CustomStrategyDataModel(domain=_make_domain())
        result = strategies_api.map(dm)
        assert isinstance(result, _DecoratedStrategy)

        # cleanup
        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)

    def test_register_updates_stepwise_strategy(self):
        """Registering a strategy should update the StepwiseStrategy data model."""
        import bofire.strategies.api as strategies_api
        from bofire.data_models.strategies.actual_strategy_type import (
            _ACTUAL_STRATEGY_TYPES,
        )
        from bofire.data_models.strategies.stepwise.stepwise import Step
        from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP

        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)
        if _CustomStrategyDataModel in _ACTUAL_STRATEGY_TYPES:
            _ACTUAL_STRATEGY_TYPES.remove(_CustomStrategyDataModel)

        strategies_api.register(_CustomStrategyDataModel, _CustomStrategy)

        # Step.strategy_data should now accept the custom type
        step = Step(
            strategy_data=_CustomStrategyDataModel(domain=_make_domain()),
            condition={"type": "AlwaysTrueCondition"},
        )
        assert type(step.strategy_data) is _CustomStrategyDataModel

        # cleanup
        ACTUAL_MAP.pop(_CustomStrategyDataModel, None)
        if _CustomStrategyDataModel in _ACTUAL_STRATEGY_TYPES:
            _ACTUAL_STRATEGY_TYPES.remove(_CustomStrategyDataModel)

    def test_register_exported_from_api(self):
        from bofire.strategies.api import register

        assert callable(register)


class TestRegisterSurrogate:
    def test_register_and_map(self):
        import bofire.surrogates.api as surrogates_api
        from bofire.surrogates.mapper import SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)

        surrogates_api.register(_CustomSurrogateDataModel, _CustomSurrogate)
        assert SURROGATE_MAP[_CustomSurrogateDataModel] is _CustomSurrogate

        dm = _CustomSurrogateDataModel()
        result = surrogates_api.map(dm)
        assert isinstance(result, _CustomSurrogate)

        # cleanup
        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)

    def test_register_with_data_model_transform(self):
        import bofire.surrogates.api as surrogates_api
        from bofire.surrogates.mapper import DATA_MODEL_MAP, SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomSurrogateDataModel, None)

        transform_called_with = []

        def my_transform(dm):
            transform_called_with.append(dm)
            # Return the same data model (identity transform)
            return dm

        surrogates_api.register(
            _CustomSurrogateDataModel,
            _CustomSurrogate,
            data_model_transform=my_transform,
        )
        assert SURROGATE_MAP[_CustomSurrogateDataModel] is _CustomSurrogate
        assert DATA_MODEL_MAP[_CustomSurrogateDataModel] is my_transform

        dm = _CustomSurrogateDataModel()
        result = surrogates_api.map(dm)
        assert isinstance(result, _CustomSurrogate)
        assert len(transform_called_with) == 1
        assert transform_called_with[0] is dm

        # cleanup
        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomSurrogateDataModel, None)

    def test_register_without_data_model_transform_does_not_add_to_data_model_map(
        self,
    ):
        import bofire.surrogates.api as surrogates_api
        from bofire.surrogates.mapper import DATA_MODEL_MAP, SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomSurrogateDataModel, None)

        surrogates_api.register(_CustomSurrogateDataModel, _CustomSurrogate)
        assert _CustomSurrogateDataModel not in DATA_MODEL_MAP

        # cleanup
        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)

    def test_register_decorator_syntax(self):
        import bofire.surrogates.api as surrogates_api
        from bofire.surrogates.mapper import SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)

        @surrogates_api.register(_CustomSurrogateDataModel)
        class _DecoratedSurrogate(_CustomSurrogate):
            pass

        assert SURROGATE_MAP[_CustomSurrogateDataModel] is _DecoratedSurrogate

        dm = _CustomSurrogateDataModel()
        result = surrogates_api.map(dm)
        assert isinstance(result, _DecoratedSurrogate)

        # cleanup
        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)

    def test_register_decorator_with_transform(self):
        import bofire.surrogates.api as surrogates_api
        from bofire.surrogates.mapper import DATA_MODEL_MAP, SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomSurrogateDataModel, None)

        transform_called = []

        def my_transform(dm):
            transform_called.append(dm)
            return dm

        @surrogates_api.register(
            _CustomSurrogateDataModel, data_model_transform=my_transform
        )
        class _DecoratedSurrogate(_CustomSurrogate):
            pass

        assert SURROGATE_MAP[_CustomSurrogateDataModel] is _DecoratedSurrogate
        assert DATA_MODEL_MAP[_CustomSurrogateDataModel] is my_transform

        dm = _CustomSurrogateDataModel()
        result = surrogates_api.map(dm)
        assert isinstance(result, _DecoratedSurrogate)
        assert len(transform_called) == 1

        # cleanup
        SURROGATE_MAP.pop(_CustomSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomSurrogateDataModel, None)

    def test_register_exported_from_api(self):
        from bofire.surrogates.api import register

        assert callable(register)


# ---------------------------------------------------------------------------
# Stub for botorch surrogate registration
# ---------------------------------------------------------------------------


class _CustomBotorchSurrogateDataModel(BotorchSurrogate):
    type: Literal["_CustomBotorchSurrogate"] = "_CustomBotorchSurrogate"

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        return True


class _CustomBotorchSurrogate(Surrogate):
    def __init__(self, data_model, **kwargs):
        self.data_model = data_model

    def predict(self, X):
        pass

    def _predict(self, transformed_X):
        pass

    def loads(self, data):
        pass

    def dumps(self):
        return ""

    def _dumps(self):
        return ""


class TestRegisterBotorchSurrogate:
    def _cleanup(self):
        from bofire.data_models.surrogates.botorch_surrogates import (
            _BOTORCH_SURROGATE_TYPES,
        )
        from bofire.surrogates.mapper import DATA_MODEL_MAP, SURROGATE_MAP

        SURROGATE_MAP.pop(_CustomBotorchSurrogateDataModel, None)
        DATA_MODEL_MAP.pop(_CustomBotorchSurrogateDataModel, None)
        if _CustomBotorchSurrogateDataModel in _BOTORCH_SURROGATE_TYPES:
            _BOTORCH_SURROGATE_TYPES.remove(_CustomBotorchSurrogateDataModel)

    def test_register_adds_to_botorch_surrogates(self):
        """Registering a BotorchSurrogate subclass should also update
        AnyBotorchSurrogate so that BotorchSurrogates accepts it."""
        import typing

        import bofire.surrogates.api as surrogates_api
        from bofire.data_models.surrogates.botorch_surrogates import (
            _BOTORCH_SURROGATE_TYPES,
            BotorchSurrogates,
        )

        self._cleanup()
        n_before = len(_BOTORCH_SURROGATE_TYPES)

        surrogates_api.register(
            _CustomBotorchSurrogateDataModel, _CustomBotorchSurrogate
        )

        # type was appended to the registry
        assert _CustomBotorchSurrogateDataModel in _BOTORCH_SURROGATE_TYPES
        assert len(_BOTORCH_SURROGATE_TYPES) == n_before + 1

        # BotorchSurrogates now accepts our custom surrogate
        dm = _CustomBotorchSurrogateDataModel(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
        )
        bs = BotorchSurrogates(surrogates=[dm])
        assert len(bs.surrogates) == 1
        assert isinstance(bs.surrogates[0], _CustomBotorchSurrogateDataModel)

        # the module-level AnyBotorchSurrogate union includes our type
        from bofire.data_models.surrogates import botorch_surrogates

        args = typing.get_args(botorch_surrogates.AnyBotorchSurrogate)
        assert _CustomBotorchSurrogateDataModel in args

        self._cleanup()

    def test_register_idempotent(self):
        """Calling register twice with the same type should not duplicate it."""
        import bofire.surrogates.api as surrogates_api
        from bofire.data_models.surrogates.botorch_surrogates import (
            _BOTORCH_SURROGATE_TYPES,
        )

        self._cleanup()

        surrogates_api.register(
            _CustomBotorchSurrogateDataModel, _CustomBotorchSurrogate
        )
        surrogates_api.register(
            _CustomBotorchSurrogateDataModel, _CustomBotorchSurrogate
        )

        count = _BOTORCH_SURROGATE_TYPES.count(_CustomBotorchSurrogateDataModel)
        assert count == 1

        self._cleanup()

    def test_map_round_trip(self):
        """A registered botorch surrogate should be mappable via surrogates.map."""
        import bofire.surrogates.api as surrogates_api

        self._cleanup()

        surrogates_api.register(
            _CustomBotorchSurrogateDataModel, _CustomBotorchSurrogate
        )

        dm = _CustomBotorchSurrogateDataModel(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
        )
        result = surrogates_api.map(dm)
        assert isinstance(result, _CustomBotorchSurrogate)

        self._cleanup()

    def test_register_decorator_syntax(self):
        """Decorator syntax should also trigger botorch registration."""
        import bofire.surrogates.api as surrogates_api
        from bofire.data_models.surrogates.botorch_surrogates import (
            _BOTORCH_SURROGATE_TYPES,
        )

        self._cleanup()

        @surrogates_api.register(_CustomBotorchSurrogateDataModel)
        class _DecoratedBotorchSurrogate(_CustomBotorchSurrogate):
            pass

        assert _CustomBotorchSurrogateDataModel in _BOTORCH_SURROGATE_TYPES

        dm = _CustomBotorchSurrogateDataModel(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
        )
        result = surrogates_api.map(dm)
        assert isinstance(result, _DecoratedBotorchSurrogate)

        self._cleanup()


# ---------------------------------------------------------------------------
# Stub data models for kernels and priors
# ---------------------------------------------------------------------------


class _CustomKernelDataModel(KernelDataModel):
    type: str = "CustomKernel"


class _CustomPriorDataModel(PriorDataModel):
    type: str = "CustomPrior"


# ---------------------------------------------------------------------------
# Kernel registration tests
# ---------------------------------------------------------------------------


class TestRegisterKernel:
    def test_register_and_map(self):
        import bofire.kernels.api as kernels_api
        from bofire.kernels.mapper import KERNEL_MAP

        KERNEL_MAP.pop(_CustomKernelDataModel, None)

        sentinel = MagicMock(name="gpytorch_kernel")

        def my_map_fn(data_model, batch_shape, active_dims, features_to_idx_mapper):
            return sentinel

        kernels_api.register(_CustomKernelDataModel, my_map_fn)
        assert KERNEL_MAP[_CustomKernelDataModel] is my_map_fn

        import torch

        dm = _CustomKernelDataModel()
        result = kernels_api.map(dm, torch.Size(), [0], None)
        assert result is sentinel

        # cleanup
        KERNEL_MAP.pop(_CustomKernelDataModel, None)

    def test_register_decorator_syntax(self):
        import bofire.kernels.api as kernels_api
        from bofire.kernels.mapper import KERNEL_MAP

        KERNEL_MAP.pop(_CustomKernelDataModel, None)

        sentinel = MagicMock(name="gpytorch_kernel")

        @kernels_api.register(_CustomKernelDataModel)
        def my_map_fn(data_model, batch_shape, active_dims, features_to_idx_mapper):
            return sentinel

        assert KERNEL_MAP[_CustomKernelDataModel] is my_map_fn

        import torch

        dm = _CustomKernelDataModel()
        result = kernels_api.map(dm, torch.Size(), [0], None)
        assert result is sentinel

        # cleanup
        KERNEL_MAP.pop(_CustomKernelDataModel, None)

    def test_register_exported_from_api(self):
        from bofire.kernels.api import register

        assert callable(register)


# ---------------------------------------------------------------------------
# Prior registration tests
# ---------------------------------------------------------------------------


class TestRegisterPrior:
    def test_register_and_map(self):
        import bofire.priors.api as priors_api
        from bofire.priors.mapper import PRIOR_MAP

        PRIOR_MAP.pop(_CustomPriorDataModel, None)

        sentinel = MagicMock(name="gpytorch_prior")

        def my_map_fn(data_model, **kwargs):
            return sentinel

        priors_api.register(_CustomPriorDataModel, my_map_fn)
        assert PRIOR_MAP[_CustomPriorDataModel] is my_map_fn

        dm = _CustomPriorDataModel()
        result = priors_api.map(dm)
        assert result is sentinel

        # cleanup
        PRIOR_MAP.pop(_CustomPriorDataModel, None)

    def test_register_decorator_syntax(self):
        import bofire.priors.api as priors_api
        from bofire.priors.mapper import PRIOR_MAP

        PRIOR_MAP.pop(_CustomPriorDataModel, None)

        sentinel = MagicMock(name="gpytorch_prior")

        @priors_api.register(_CustomPriorDataModel)
        def my_map_fn(data_model, **kwargs):
            return sentinel

        assert PRIOR_MAP[_CustomPriorDataModel] is my_map_fn

        dm = _CustomPriorDataModel()
        result = priors_api.map(dm)
        assert result is sentinel

        # cleanup
        PRIOR_MAP.pop(_CustomPriorDataModel, None)

    def test_register_exported_from_api(self):
        from bofire.priors.api import register

        assert callable(register)


# ---------------------------------------------------------------------------
# Integration tests: custom types accepted by Pydantic validation
# ---------------------------------------------------------------------------


class _IntegrationKernelDataModel(KernelDataModel):
    type: Literal["_IntegrationKernel"] = "_IntegrationKernel"
    my_param: float = 1.0


class _IntegrationContinuousKernel(_ContinuousBase):
    type: Literal["_IntegrationContinuousKernel"] = "_IntegrationContinuousKernel"


class _IntegrationPriorDataModel(PriorDataModel):
    type: Literal["_IntegrationPrior"] = "_IntegrationPrior"
    value: float = 1.0


class TestKernelPydanticIntegration:
    """After register_kernel, custom kernel types should pass Pydantic validation
    in surrogate models and aggregation kernels."""

    def test_custom_kernel_in_surrogate(self):
        from bofire.data_models.kernels.api import register_kernel

        register_kernel(_IntegrationKernelDataModel)

        from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate

        s = SingleTaskGPSurrogate(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
            kernel=_IntegrationKernelDataModel(my_param=42.0),
        )
        assert isinstance(s.kernel, _IntegrationKernelDataModel)
        assert s.kernel.my_param == 42.0

    def test_custom_kernel_in_additive_kernel(self):
        from bofire.data_models.kernels.aggregation import AdditiveKernel
        from bofire.data_models.kernels.api import register_kernel
        from bofire.data_models.kernels.continuous import RBFKernel

        register_kernel(_IntegrationKernelDataModel)

        ak = AdditiveKernel(
            kernels=[RBFKernel(), _IntegrationKernelDataModel(my_param=7.0)]
        )
        assert len(ak.kernels) == 2
        assert isinstance(ak.kernels[1], _IntegrationKernelDataModel)

    def test_custom_kernel_in_scale_kernel(self):
        from bofire.data_models.kernels.aggregation import ScaleKernel
        from bofire.data_models.kernels.api import register_kernel

        register_kernel(_IntegrationKernelDataModel)

        sk = ScaleKernel(base_kernel=_IntegrationKernelDataModel(my_param=3.0))
        assert isinstance(sk.base_kernel, _IntegrationKernelDataModel)

    def test_custom_continuous_kernel_in_mixed_surrogate(self):
        """A ContinuousKernel subclass should be auto-added to AnyContinuousKernel."""
        from bofire.data_models.kernels.api import (
            _CONTINUOUS_KERNEL_TYPES,
            register_kernel,
        )
        from bofire.data_models.surrogates.mixed_single_task_gp import (
            MixedSingleTaskGPSurrogate,
        )

        register_kernel(_IntegrationContinuousKernel)

        assert _IntegrationContinuousKernel in _CONTINUOUS_KERNEL_TYPES

        s = MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    CategoricalInput(key="c", categories=["x", "y"]),
                ]
            ),
            outputs=_OUTPUTS,
            continuous_kernel=_IntegrationContinuousKernel(),
        )
        assert isinstance(s.continuous_kernel, _IntegrationContinuousKernel)

    def test_mapper_register_also_updates_pydantic(self):
        """The mapper-level register() should trigger data model registration."""
        import bofire.kernels.api as kernels_api
        from bofire.data_models.kernels.api import _KERNEL_TYPES

        # Use a fresh class to ensure it's not already registered
        class _MapperKernel(KernelDataModel):
            type: Literal["_MapperKernel"] = "_MapperKernel"

        sentinel = MagicMock(name="gpytorch_kernel")

        kernels_api.register(_MapperKernel, lambda *a: sentinel)

        assert _MapperKernel in _KERNEL_TYPES

        from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate

        s = SingleTaskGPSurrogate(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
            kernel=_MapperKernel(),
        )
        assert isinstance(s.kernel, _MapperKernel)


class TestPriorPydanticIntegration:
    """After register_prior, custom prior types should pass Pydantic validation
    in kernel and surrogate model fields."""

    def test_custom_prior_as_noise_prior(self):
        from bofire.data_models.priors.api import register_prior

        register_prior(_IntegrationPriorDataModel)

        from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate

        s = SingleTaskGPSurrogate(
            inputs=_INPUTS,
            outputs=_OUTPUTS,
            noise_prior=_IntegrationPriorDataModel(value=2.0),
        )
        assert isinstance(s.noise_prior, _IntegrationPriorDataModel)
        assert s.noise_prior.value == 2.0

    def test_custom_prior_as_lengthscale_prior(self):
        from bofire.data_models.kernels.continuous import RBFKernel
        from bofire.data_models.priors.api import register_prior

        register_prior(_IntegrationPriorDataModel)

        k = RBFKernel(lengthscale_prior=_IntegrationPriorDataModel(value=3.0))
        assert isinstance(k.lengthscale_prior, _IntegrationPriorDataModel)

    def test_mapper_register_also_updates_pydantic(self):
        """The mapper-level register() should trigger data model registration."""
        import bofire.priors.api as priors_api
        from bofire.data_models.priors.api import _PRIOR_TYPES

        class _MapperPrior(PriorDataModel):
            type: Literal["_MapperPrior"] = "_MapperPrior"

        priors_api.register(_MapperPrior, lambda dm, **kw: MagicMock())

        assert _MapperPrior in _PRIOR_TYPES

        from bofire.data_models.kernels.continuous import RBFKernel

        k = RBFKernel(lengthscale_prior=_MapperPrior())
        assert isinstance(k.lengthscale_prior, _MapperPrior)


# ---------------------------------------------------------------------------
# Engineered feature registration tests
# ---------------------------------------------------------------------------


class _IntegrationEngineeredFeature(EngineeredFeature):
    type: Literal["_IntegrationEngineered"] = "_IntegrationEngineered"
    order_id = 99

    @property
    def n_transformed_inputs(self) -> int:
        return 1


class TestEngineeredFeatureRegistration:
    def test_register_data_model(self):
        from bofire.data_models.domain.features import EngineeredFeatures
        from bofire.data_models.features.api import register_engineered_feature

        register_engineered_feature(_IntegrationEngineeredFeature)

        ef = EngineeredFeatures(
            features=[_IntegrationEngineeredFeature(key="test", features=["a", "b"])]
        )
        assert isinstance(ef.features[0], _IntegrationEngineeredFeature)

    def test_register_in_surrogate(self):
        from bofire.data_models.domain.features import EngineeredFeatures
        from bofire.data_models.features.api import (
            ContinuousInput,
            register_engineered_feature,
        )
        from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate

        register_engineered_feature(_IntegrationEngineeredFeature)

        ef = EngineeredFeatures(
            features=[_IntegrationEngineeredFeature(key="test", features=["a", "b"])]
        )
        s = SingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                ]
            ),
            outputs=_OUTPUTS,
            engineered_features=ef,
        )
        assert len(s.engineered_features.features) == 1

    def test_mapper_register_decorator(self):
        from bofire.data_models.features.api import _ENGINEERED_FEATURE_TYPES
        from bofire.surrogates.engineered_features import AGGREGATE_MAP, register

        class _MapperEngineered(EngineeredFeature):
            type: Literal["_MapperEngineered"] = "_MapperEngineered"
            order_id = 100

            @property
            def n_transformed_inputs(self) -> int:
                return 1

        sentinel = MagicMock(name="append_features")

        @register(_MapperEngineered)
        def my_map_fn(inputs, transform_specs, feature):
            return sentinel

        assert AGGREGATE_MAP[_MapperEngineered] is my_map_fn
        assert _MapperEngineered in _ENGINEERED_FEATURE_TYPES

    def test_mapper_register_direct_call(self):
        from bofire.data_models.features.api import _ENGINEERED_FEATURE_TYPES
        from bofire.surrogates.engineered_features import AGGREGATE_MAP, register

        class _DirectEngineered(EngineeredFeature):
            type: Literal["_DirectEngineered"] = "_DirectEngineered"
            order_id = 101

            @property
            def n_transformed_inputs(self) -> int:
                return 1

        sentinel = MagicMock(name="append_features")
        register(_DirectEngineered, lambda i, t, f: sentinel)

        assert _DirectEngineered in AGGREGATE_MAP
        assert _DirectEngineered in _ENGINEERED_FEATURE_TYPES


# ---------------------------------------------------------------------------
# Introspection test: ensure _rebuild_dependent_models covers all fields
# ---------------------------------------------------------------------------


class TestRebuildCoverage:
    """Verify that the explicit field lists in _rebuild_dependent_models
    cover every Pydantic model field typed with AnyPrior, AnyPriorConstraint,
    or AnyKernel.  This catches regressions when new surrogates or kernels
    are added without updating the rebuild functions."""

    @staticmethod
    def _collect_fields(base_module: str, target_union_names: set[str]):
        """Walk all Pydantic models under *base_module* and return
        ``{(ModelClass, field_name)}`` for every field whose annotation
        string matches one of the *target_union_names*.
        """
        import importlib
        import inspect
        import pkgutil

        import pydantic

        pkg = importlib.import_module(base_module)
        found: set[tuple[type, str]] = set()

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + "."
        ):
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            for _name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    not issubclass(obj, pydantic.BaseModel)
                    or obj is pydantic.BaseModel
                    or obj.__module__ != modname
                ):
                    continue
                for field_name, field_info in obj.model_fields.items():
                    ann = field_info.annotation
                    ann_str = str(ann)
                    for target in target_union_names:
                        if target in ann_str:
                            found.add((obj, field_name))
        return found

    def test_prior_rebuild_covers_all_anyprior_fields(self):
        """Every model field typed as AnyPrior should appear in
        priors._rebuild_dependent_models."""
        import bofire.data_models.priors.api

        src = inspect.getsource(bofire.data_models.priors.api._rebuild_dependent_models)

        patched: set[tuple[str, str]] = set()

        # The function calls patch_field(Model, "field", union) — extract those pairs
        import re

        for match in re.finditer(r"\((\w+),\s*\"(\w+)\"\)", src):
            model_name, field_name = match.group(1), match.group(2)
            patched.add((model_name, field_name))

        # Now collect all fields in the codebase typed with AnyPrior or AnyPriorConstraint
        all_fields = self._collect_fields(
            "bofire.data_models", {"AnyPrior", "AnyPriorConstraint"}
        )

        # Convert to (class_name, field_name) for comparison
        all_field_names = {(cls.__name__, fname) for cls, fname in all_fields}

        missing = all_field_names - patched
        assert not missing, (
            f"Fields typed as AnyPrior/AnyPriorConstraint not covered by "
            f"_rebuild_dependent_models: {missing}"
        )

    def test_kernel_rebuild_covers_all_anykernel_fields(self):
        """Every model field typed as AnyKernel should appear in
        kernels._rebuild_dependent_models or be handled via
        append_to_union_field."""
        import bofire.data_models.kernels.api

        src = inspect.getsource(
            bofire.data_models.kernels.api._rebuild_dependent_models
        )

        import re

        patched: set[tuple[str, str]] = set()
        for match in re.finditer(r"\((\w+),\s*\"(\w+)\"\)", src):
            model_name, field_name = match.group(1), match.group(2)
            patched.add((model_name, field_name))

        all_fields = self._collect_fields("bofire.data_models", {"AnyKernel"})
        all_field_names = {(cls.__name__, fname) for cls, fname in all_fields}

        # AnyContinuousKernel / AnyCategoricalKernel are handled separately
        # so exclude them from this check
        continuous_cat_fields = self._collect_fields(
            "bofire.data_models", {"AnyContinuousKernel", "AnyCategoricalKernel"}
        )
        continuous_cat_names = {
            (cls.__name__, fname) for cls, fname in continuous_cat_fields
        }

        # Only check pure AnyKernel fields
        anykernel_only = all_field_names - continuous_cat_names
        missing = anykernel_only - patched
        assert not missing, (
            f"Fields typed as AnyKernel not covered by "
            f"_rebuild_dependent_models: {missing}"
        )
