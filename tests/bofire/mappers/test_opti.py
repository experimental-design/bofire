import pytest
from opti import Problem
from opti.problems.cbo_benchmarks import (
    G4,
    G6,
    G7,
    G8,
    G9,
    G10,
    Gardner,
    Gramacy,
    PressureVessel,
    Sasena,
    SpeedReducer,
    TensionCompression,
    WeldedBeam1,
)
from opti.problems.datasets import (
    HPLC,
    Alkox,
    BaumgartnerAniline,
    BaumgartnerBenzamide,
    Benzylation,
    Cake,
    Fullerenes,
    Photodegradation,
    ReizmanSuzuki,
    SnAr,
    Suzuki,
)
from opti.problems.detergent import (
    Detergent,
    Detergent_NChooseKConstraint,
    Detergent_OutputConstraint,
    Detergent_TwoOutputConstraints,
)
from opti.problems.mixed import DiscreteFuelInjector, DiscreteVLMOP2
from opti.problems.multi import (
    Daechert1,
    Daechert2,
    Daechert3,
    Hyperellipsoid,
    OmniTest,
    Poloni,
    Qapi1,
    WeldedBeam,
)
from opti.problems.single import (
    Ackley,
    Branin,
    Himmelblau,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    ThreeHumpCamel,
    Zakharov,
    Zakharov_Categorical,
    Zakharov_Constrained,
    Zakharov_NChooseKConstraint,
)
from opti.problems.univariate import Line1D, Parabola1D, Sigmoid1D, Sinus1D, Step1D
from opti.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from pandas.testing import assert_frame_equal

from bofire.mappers.opti import domain2problem, problem2domain


@pytest.mark.parametrize(
    "problem",
    [
        Ackley(),
        Branin(),
        Himmelblau(),
        Michalewicz(),
        Rastrigin(),
        Rosenbrock(),
        Schwefel(),
        Sphere(),
        ThreeHumpCamel(),
        Zakharov(),
        Zakharov_Categorical(),
        Zakharov_Constrained(),
        Zakharov_NChooseKConstraint(),
        G4(),
        G6(),
        G7(),
        G8(),
        G9(),
        G10(),
        Gardner(),
        Gramacy(),
        PressureVessel(),
        Sasena(),
        SpeedReducer(),
        TensionCompression(),
        WeldedBeam1(),
        HPLC(),
        Alkox(),
        BaumgartnerAniline(),
        BaumgartnerBenzamide(),
        Benzylation(),
        Cake(),
        Fullerenes(),
        Photodegradation(),
        ReizmanSuzuki(),
        SnAr(),
        Suzuki(),
        Detergent(),
        Detergent_NChooseKConstraint(),
        Detergent_OutputConstraint(),
        Detergent_TwoOutputConstraints(),
        DiscreteFuelInjector(),
        DiscreteVLMOP2(),
        Daechert1(),
        Daechert2(),
        Daechert3(),
        Hyperellipsoid(),
        OmniTest(),
        Poloni(),
        Qapi1(),
        WeldedBeam(),
        Line1D(),
        Parabola1D(),
        Sigmoid1D(),
        Sinus1D(),
        Step1D(),
        ZDT1(),
        ZDT2(),
        ZDT3(),
        ZDT4(),
        ZDT6(),
    ],
)
def test_problem2domain(problem: Problem):
    # go from problem to config to domain to config2 to problem2 and domain2
    config = problem.to_config()
    domain = problem2domain(config)
    config2 = domain2problem(domain)
    problem2 = Problem.from_config(config2)
    domain2 = problem2domain(config2)

    # compare domains
    assert sorted(domain2.input_features) == sorted(domain.input_features)
    assert domain2.constraints == domain.constraints
    assert sorted(domain2.output_features) == sorted(domain2.output_features)
    if "data" in config:
        assert_frame_equal(
            domain2.experiments, domain.experiments, check_like=True, check_dtype=False
        )

    # compare problems
    assert config2["inputs"] == config["inputs"]
    assert config2["objectives"] == config["objectives"]
    if "constraints" in config:
        assert config["constraints"] == config2["constraints"]
    if "data" in config:
        assert_frame_equal(
            problem2.data, problem.data, check_like=True, check_dtype=False
        )
