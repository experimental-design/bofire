import inspect
import types
import typing
from typing import get_origin, get_type_hints

import typing_extensions

import bofire.data_models.strategies.api as dms
from bofire.benchmarks.single import MultiTaskHimmelblau
from bofire.data_models.features.task import TaskInput
from bofire.strategies.api import (
    AdditiveSoboStrategy,
    DoEStrategy,
    EntingStrategy,
    MoboStrategy,
    MultiFidelityStrategy,
    MultiplicativeAdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QparegoStrategy,
    RandomStrategy,
    SoboStrategy,
)
from bofire.strategies.fractional_factorial import FractionalFactorialStrategy
from tests.bofire.strategies.test_base import domains


def remove_optional(anno):
    origin = get_origin(anno)
    if origin == typing.Union or origin is types.UnionType:
        return sorted((a for a in anno.__args__ if a is not type(None)), key=hash)
    else:
        return [anno]


def test_make():
    so_domain = domains[0]
    mo_domain = domains[4]
    benchmark = MultiTaskHimmelblau()
    (task_input,) = benchmark.domain.inputs.get(TaskInput, exact=True)
    assert task_input.type == "TaskInput"
    task_input.fidelities = [0, 1]
    mt_domain = benchmark.domain

    def test(strat, dm, domain):
        data_model = strat.data_model_cls(domain=domain)
        data_model_dump = data_model.model_dump()

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
            ), f"{k} not in {strat.__name__}'s make parameters"

        # do the non-optional annotation-parts match?
        for name, p_annotation in get_type_hints(strat.make).items():
            if name == "return":
                assert p_annotation is typing_extensions.Self
            else:
                dm_anno = dm.model_fields[name].annotation
                p_anno = p_annotation

                dm_anno = remove_optional(dm_anno)
                p_anno = remove_optional(p_anno)
                assert (
                    len(dm_anno) == len(p_anno)
                ), f"{strat.__name__}. Annotations do not match for {name}: {dm_anno} !=\n {p_anno}"
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
                    ), f"{strat.__name__}. Annotations do not match for {name}: {da} !=\n {pa}"

        strat1 = strat(data_model=data_model)
        strat2 = strat.make(domain=domain)

        data_model = dm(domain=domain)
        strat3 = strat(data_model=data_model)

        assert strat1.data_model == strat2.data_model
        assert strat1.data_model == strat3.data_model

    test(SoboStrategy, dms.SoboStrategy, so_domain)
    test(AdditiveSoboStrategy, dms.AdditiveSoboStrategy, mo_domain)
    test(MultiplicativeSoboStrategy, dms.MultiplicativeSoboStrategy, mo_domain)
    test(
        MultiplicativeAdditiveSoboStrategy,
        dms.MultiplicativeAdditiveSoboStrategy,
        mo_domain,
    )
    test(QparegoStrategy, dms.QparegoStrategy, mo_domain)
    test(MultiFidelityStrategy, dms.MultiFidelityStrategy, mt_domain)
    test(MoboStrategy, dms.MoboStrategy, mo_domain)
    test(EntingStrategy, dms.EntingStrategy, so_domain)
    test(DoEStrategy, dms.DoEStrategy, so_domain)
    test(FractionalFactorialStrategy, dms.FractionalFactorialStrategy, so_domain)
    test(RandomStrategy, dms.RandomStrategy, so_domain)


if __name__ == "__main__":
    test_make()
