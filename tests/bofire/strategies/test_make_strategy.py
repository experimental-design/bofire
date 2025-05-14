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


def test_make_default():
    so_domain = domains[0]
    mo_domain = domains[4]
    benchmark = MultiTaskHimmelblau()
    (task_input,) = benchmark.domain.inputs.get(TaskInput, exact=True)
    assert task_input.type == "TaskInput"
    task_input.fidelities = [0, 1]
    mt_domain = benchmark.domain

    def test(strat, dm, domain):
        data_model = strat.data_model_cls(domain=domain)
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
    test_make_default()
