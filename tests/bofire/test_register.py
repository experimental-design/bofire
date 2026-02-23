from typing import Literal, Type

import pandas as pd

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyOutput,
    ContinuousInput,
    ContinuousOutput,
    Feature,
)
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

    def test_register_meta_strategy(self):
        import bofire.strategies.api as strategies_api
        from bofire.strategies.mapper_meta import STRATEGY_MAP as META_MAP

        META_MAP.pop(_CustomStrategyDataModel, None)

        strategies_api.register(_CustomStrategyDataModel, _CustomStrategy, meta=True)
        assert META_MAP[_CustomStrategyDataModel] is _CustomStrategy

        # meta strategies are checked first in map()
        dm = _CustomStrategyDataModel(domain=_make_domain())
        result = strategies_api.map(dm)
        assert isinstance(result, _CustomStrategy)

        # cleanup
        META_MAP.pop(_CustomStrategyDataModel, None)

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
