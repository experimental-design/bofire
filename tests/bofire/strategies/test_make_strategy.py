import inspect
import types
import typing
from typing import get_origin, get_type_hints

import typing_extensions

import bofire.strategies.api as strats
from bofire.strategies.factorial import FactorialStrategy
from tests.bofire.data_models.specs.api import strategies as strat_specs


def remove_optional(anno):
    origin = get_origin(anno)
    if origin == typing.Union or origin is types.UnionType:
        return sorted((a for a in anno.__args__ if a is not type(None)), key=hash)
    else:
        return [anno]


def test_make():
    def test(strat, data_model):
        data_model_dump = data_model.model_dump()
        data_model_dump.pop("type")

        sig = inspect.signature(strat.make)
        param_names_make = list(sig.parameters.keys())

        # are all make parameters in the data model?
        for arg_name in param_names_make:
            assert (
                arg_name in data_model_dump
            ), f"Missing {arg_name} in {strat.__name__}'s data model"

        # are all data model parameters in the make function?
        data_model_field_names = [k for k in data_model_dump.keys() if k != "type"]
        for k in data_model_field_names:
            assert (
                k in param_names_make
            ), f"{k} not in {type(strat).__name__}'s make parameters"

        # do the non-optional annotation-parts match?
        for name, p_annotation in get_type_hints(strat.make).items():
            if name == "return":
                assert p_annotation is typing_extensions.Self
            else:
                dm_anno = type(data_model).model_fields[name].annotation
                p_anno = p_annotation

                dm_anno = remove_optional(dm_anno)
                p_anno = remove_optional(p_anno)
                assert (
                    len(dm_anno) == len(p_anno)
                ), f"{type(strat).__name__}. Annotations do not match for {name}: {dm_anno} !=\n {p_anno}"
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
                    ), f"{type(strat).__name__}. Annotations do not match for {name}: {da} !=\n {pa}"

        made_strat = strat.make(**data_model_dump)
        made_dump = made_strat._data_model.model_dump()
        made_dump.pop("type")
        assert len(made_dump) == len(data_model_dump)
        for k, made_v in made_dump.items():
            v = data_model_dump[k]
            assert made_v == v, f"{strat.__name__}. {k} does not match: {made_v} != {v}"

    strats_without_make = [
        FactorialStrategy,  # this is deprecated in favor of F, hence we don't add make
    ]
    for spec in strat_specs.valids:
        data_model = spec.obj()
        strat = strats.map(data_model)
        if type(strat) not in strats_without_make:
            # test the make function
            test(strat, data_model)
        else:
            print(f"Skipping {strat} because it does not have a make function")


if __name__ == "__main__":
    test_make()
