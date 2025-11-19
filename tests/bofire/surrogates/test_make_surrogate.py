import inspect
import types
import typing
from typing import get_origin, get_type_hints, Protocol

import typing_extensions

from bofire.data_models.surrogates.bnn import SingleTaskIBNNSurrogate
from bofire.data_models.surrogates.mixed_single_task_gp import MixedSingleTaskGPSurrogate
from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
import bofire.surrogates.api as surrogates
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.shape import PiecewiseLinearGPSurrogate
from bofire.surrogates.surrogate import Surrogate
from tests.bofire.data_models.specs.api import surrogates as surrogate_specs

surrogates_skip_annotations = [
    SingleTaskIBNNSurrogate,
    RandomForestSurrogate,
    MultiTaskGPSurrogate
]

surrogates_skip_all = [
    MixedSingleTaskGPSurrogate,
    RegressionMLPEnsemble,
    ClassificationMLPEnsemble,
    PiecewiseLinearGPSurrogate
]

def remove_optional(anno):
    origin = get_origin(anno)
    if origin == typing.Union or origin is types.UnionType:
        return sorted((a for a in anno.__args__ if a is not type(None)), key=hash)
    else:
        return [anno]


def test_make():
    class SurrogateWithMake(Protocol):
        @classmethod
        def make(cls, *args, **kwargs) -> Surrogate: ...

    def test(surrogate_type: type[SurrogateWithMake], data_model):
        if (
            type(data_model) in surrogates_skip_all
            or type(surrogate_type) in surrogates_skip_all
        ):
            return
        
        data_model_dump = data_model.model_dump()
        data_model_dump.pop("type")

        sig = inspect.signature(surrogate_type.make)
        param_names_make = list(sig.parameters.keys())

        # are all make parameters in the data model?
        for arg_name in param_names_make:
            assert (
                arg_name in data_model_dump
            ), f"Missing {arg_name} in {type(data_model)}'s data model"

        # are all data model parameters in the make function?
        data_model_field_names = [k for k in data_model_dump.keys() if k != "type"]
        for k in data_model_field_names:
            assert (
                k in param_names_make
            ), f"{k} not in {type(surrogate_type).__name__}'s make parameters"

        if not type(data_model) in surrogates_skip_annotations:        
            # do the non-optional annotation-parts match?
            for name, p_annotation in get_type_hints(surrogate_type.make).items():
                if name == "return":
                    assert p_annotation is typing_extensions.Self
                else:
                    dm_anno = type(data_model).model_fields[name].annotation
                    p_anno = p_annotation

                    dm_anno = remove_optional(dm_anno)
                    p_anno = remove_optional(p_anno)
                    assert (
                        len(dm_anno) == len(p_anno)
                    ), f"{type(surrogate_type).__name__}. Annotations do not match for {name}: {dm_anno} !=\n {p_anno}"
                    for da, pa in zip(dm_anno, p_anno):
                        if get_origin(da) == typing.Annotated:
                            da_ = da.__origin__
                        else:
                            da_ = da

                        if get_origin(pa) == typing.Annotated:
                            pa_ = pa.__origin__
                        else:
                            pa_ = pa
                        assert (
                            da_ == pa_
                        ), f"{type(surrogate_type).__name__}. Annotations do not match for {name}: {da} !=\n {pa}"

        made_surrogate = surrogate_type.make(**data_model_dump)
        for k in data_model_dump.keys():
            field_info = getattr(type(data_model), 'model_fields', {}).get(k, None)
            is_optional = False
            if field_info is not None:
                anno = field_info.annotation
                origin = get_origin(anno)
                if origin is typing.Union or origin is types.UnionType:
                    if type(None) in anno.__args__:
                        is_optional = True
            if not is_optional:
                assert hasattr(
                    made_surrogate, k
                ), f"{type(surrogate_type).__name__}. {k} missing in made_strat"

    for spec in surrogate_specs.valids:
        data_model = spec.obj()
        surrogate_type = surrogates.map(data_model)
        test(surrogate_type, data_model)

if __name__ == "__main__":
    test_make()
