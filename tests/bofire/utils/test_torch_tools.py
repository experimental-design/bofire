import random

import numpy as np
import pytest
import torch
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.utils.objective import compute_smoothed_feasibility_indicator

import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    ConstrainedCategoricalObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
)
from bofire.data_models.strategies.api import RandomStrategy
from bofire.utils.torch_tools import (
    constrained_objective2botorch,
    get_additive_botorch_objective,
    get_custom_botorch_objective,
    get_initial_conditions_generator,
    get_interpoint_constraints,
    get_linear_constraints,
    get_multiobjective_objective,
    get_multiplicative_botorch_objective,
    get_nchoosek_constraints,
    get_nonlinear_constraints,
    get_objective_callable,
    get_output_constraints,
    get_product_constraints,
    tkwargs,
)

if1 = ContinuousInput(
    bounds=(0, 1),
    key="if1",
)
if2 = ContinuousInput(
    bounds=(0, 1),
    key="if2",
)
if3 = ContinuousInput(
    bounds=(0, 1),
    key="if3",
)
if4 = ContinuousInput(
    bounds=(0.1, 0.1),
    key="if4",
)
if5 = CategoricalInput(
    categories=["a", "b", "c"],
    key="if5",
)
if6 = CategoricalInput(
    categories=["a", "b", "c"],
    allowed=[False, True, False],
    key="if6",
)
c1 = LinearEqualityConstraint(
    features=["if1", "if2", "if3", "if4"], coefficients=[1.0, 1.0, 1.0, 1.0], rhs=1.0
)
c2 = LinearInequalityConstraint(
    features=["if1", "if2"], coefficients=[1.0, 1.0], rhs=0.2
)
c3 = LinearInequalityConstraint(
    features=["if1", "if2", "if4"], coefficients=[1.0, 1.0, 0.5], rhs=0.2
)


@pytest.mark.parametrize(
    "objective",
    [
        MaximizeObjective(w=0.5),
        MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
        MinimizeObjective(w=0.5),
        MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
        TargetObjective(target_value=2.0, steepness=1.0, tolerance=1e-3, w=0.5),
        CloseToTargetObjective(target_value=2.0, exponent=1.0, w=0.5),
        # ConstantObjective(w=0.5, value=1.0),
    ],
)
def test_get_objective_callable(objective):
    samples = (torch.rand(50, 3, requires_grad=True) * 5.0).to(**tkwargs)
    a_samples = samples.detach().numpy()
    callable = get_objective_callable(idx=1, objective=objective)
    assert np.allclose(
        # objective.reward(samples, desFunc)[0].detach().numpy(),
        callable(samples).detach().numpy(),
        objective(a_samples[:, 1]),
        rtol=1e-06,
    )


def f1(samples, callables, weights, X):
    outputs_list = []
    for c, w in zip(callables, weights):
        outputs_list.append(c(samples, None) ** w)
    samples = torch.stack(outputs_list, dim=-1)

    return (samples[..., 0] + samples[..., 1]) * (samples[..., 0] * samples[..., 1])


def f2(samples, callables, weights, X):
    outputs_list = []
    for c, w in zip(callables, weights):
        outputs_list.append(c(samples, None) ** w)
    samples = torch.stack(outputs_list, dim=-1)

    return (
        (samples[..., 0] + samples[..., 1])
        * (samples[..., 0] * samples[..., 1])
        * (samples[..., 0] * samples[..., 2])
    )


@pytest.mark.parametrize("f, exclude_constraints", [(f1, True), (f2, False)])
def test_get_custom_botorch_objective(f, exclude_constraints):
    samples = (torch.rand(30, 3, requires_grad=True) * 5).to(**tkwargs)
    a_samples = samples.detach().numpy()
    obj1 = MaximizeObjective(w=1.0)
    obj2 = MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=1.0)
    obj3 = CloseToTargetObjective(w=1.0, target_value=2, exponent=1)
    outputs = Outputs(
        features=[
            ContinuousOutput(
                key="alpha",
                objective=obj1,
            ),
            ContinuousOutput(
                key="beta",
                objective=obj2,
            ),
            ContinuousOutput(
                key="gamma",
                objective=obj3,
            ),
        ]
    )
    objective = get_custom_botorch_objective(
        outputs, f=f, exclude_constraints=exclude_constraints
    )
    generic_objective = GenericMCObjective(objective=objective)
    objective_forward = generic_objective.forward(samples)

    # calc with numpy
    reward1 = obj1(a_samples[:, 0])
    reward2 = obj2(a_samples[:, 1])
    reward3 = obj3(a_samples[:, 2])
    # do the comparison
    assert np.allclose(
        (
            (reward1**obj1.w + reward3**obj3.w)
            * (reward1**obj1.w * reward3**obj3.w)
            if exclude_constraints
            else (reward1**obj1.w + reward2**obj2.w)
            * (reward1**obj1.w * reward2**obj2.w)
            * (reward1**obj1.w * reward3**obj3.w)
        ),
        objective_forward.detach().numpy(),
        rtol=1e-06,
    )
    if exclude_constraints:
        constraints, etas = get_output_constraints(outputs=outputs)
        generic_objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            eta=torch.tensor(etas).to(**tkwargs),
        )
        objective_forward = generic_objective.forward(samples)
        assert np.allclose(
            np.clip(
                (
                    (reward1**obj1.w + reward3**obj3.w)
                    * (reward1**obj1.w * reward3**obj3.w)
                )
                * reward2,
                0,
                None,
            ),
            objective_forward.detach().numpy(),
            rtol=1e-06,
        )


def test_get_multiplicative_botorch_objective():
    (obj1, obj2) = random.choices(
        [
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=1),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=2.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(target_value=2.0, exponent=1.0, w=1.0),
        ],
        k=2,
    )
    outputs = Outputs(
        features=[
            ContinuousOutput(key="alpha", objective=obj1),
            ContinuousOutput(key="beta", objective=obj2),
        ]
    )
    objective = get_multiplicative_botorch_objective(outputs)
    generic_objective = GenericMCObjective(objective=objective)
    samples = (torch.rand(30, 2, requires_grad=True) * 5).to(**tkwargs)
    a_samples = samples.detach().numpy()
    objective_forward = generic_objective.forward(samples)
    # calc with numpy
    reward1 = obj1(a_samples[:, 0])
    reward2 = obj2(a_samples[:, 1])
    # do the comparison
    assert np.allclose(
        # objective.reward(samples, desFunc)[0].detach().numpy(),
        reward1**obj1.w * reward2**obj2.w,
        objective_forward.detach().numpy(),
        rtol=1e-06,
    )


@pytest.mark.parametrize("exclude_constraints", [True, False])
def test_get_additive_botorch_objective(exclude_constraints):
    samples = (torch.rand(30, 3, requires_grad=True) * 5).to(**tkwargs)
    a_samples = samples.detach().numpy()
    obj1 = MaximizeObjective(w=0.5)
    obj2 = MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5)
    obj3 = CloseToTargetObjective(w=0.5, target_value=2, exponent=1)
    # obj3 = MaximizeObjective(w=0.7)
    outputs = Outputs(
        features=[
            ContinuousOutput(
                key="alpha",
                objective=obj1,
            ),
            ContinuousOutput(
                key="beta",
                objective=obj2,
            ),
            ContinuousOutput(
                key="gamma",
                objective=obj3,
            ),
        ]
    )
    objective = get_additive_botorch_objective(
        outputs, exclude_constraints=exclude_constraints
    )
    generic_objective = GenericMCObjective(objective=objective)
    objective_forward = generic_objective.forward(samples)

    # calc with numpy
    reward1 = obj1(a_samples[:, 0])
    reward2 = obj2(a_samples[:, 1])
    reward3 = obj3(a_samples[:, 2])
    # do the comparison
    assert np.allclose(
        # objective.reward(samples, desFunc)[0].detach().numpy(),
        (
            reward1 * obj1.w + reward3 * obj3.w
            if exclude_constraints
            else reward1 * obj1.w + reward3 * obj3.w + reward2 * obj2.w
        ),
        objective_forward.detach().numpy(),
        rtol=1e-06,
    )
    if exclude_constraints:
        constraints, etas = get_output_constraints(outputs=outputs)
        generic_objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            eta=torch.tensor(etas).to(**tkwargs),
        )
        objective_forward = generic_objective.forward(samples)
        assert np.allclose(
            np.clip((reward1 * obj1.w + reward3 * obj3.w) * reward2, 0, None),
            objective_forward.detach().numpy(),
            rtol=1e-06,
        )


def test_get_interpoint_equality_constraints():
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(1, 1)),
            ]
        ),
        constraints=Constraints(
            constraints=[
                InterpointEqualityConstraint(feature="b", multiplicity=3),
            ]
        ),
    )
    assert len(get_interpoint_constraints(domain=domain, n_candidates=9)) == 0
    domain.inputs.get_by_key("b").bounds = (0, 1)
    constraints = get_interpoint_constraints(domain=domain, n_candidates=6)
    assert len(constraints) == 4
    for c in constraints:
        assert c[2] == 0.0
        assert torch.allclose(c[1], torch.tensor([1.0, -1.0]).to(**tkwargs))
    c = constraints[0]
    assert torch.allclose(
        c[0],
        torch.tensor(
            [[0, 1], [1, 1]],
            dtype=torch.int64,
        ),
    )
    c = constraints[-1]
    assert torch.allclose(
        c[0],
        torch.tensor(
            [[3, 1], [5, 1]],
            dtype=torch.int64,
        ),
    )
    constraints = get_interpoint_constraints(domain=domain, n_candidates=8)
    assert len(constraints) == 5
    c = constraints[-1]
    assert torch.allclose(
        c[0],
        torch.tensor(
            [[6, 1], [7, 1]],
            dtype=torch.int64,
        ),
    )
    constraints = get_interpoint_constraints(domain=domain, n_candidates=3)
    assert len(constraints) == 2
    c = constraints[-1]
    assert torch.allclose(
        c[0],
        torch.tensor(
            [[0, 1], [2, 1]],
            dtype=torch.int64,
        ),
    )


def test_get_linear_constraints():
    domain = Domain(inputs=[if1, if2])
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 0

    domain = Domain(inputs=[if1, if2, if3], constraints=[c2])
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c2.rhs * -1
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1.0, -1.0]).to(**tkwargs))

    domain = Domain(
        inputs=[if1, if2, if3, if4],
        constraints=[c1, c2],
    )
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == -1 * (c1.rhs - 0.1)
    assert len(constraints[0][0]) == len(c1.features) - 1
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1, 2]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1, -1, -1]).to(**tkwargs))
    assert len(constraints[0][1]) == len(c1.coefficients) - 1
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c2.rhs * -1
    assert len(constraints[0][0]) == len(c2.features)
    assert len(constraints[0][1]) == len(c2.coefficients)
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1.0, -1.0]).to(**tkwargs))

    domain = Domain(
        inputs=[if1, if2, if3, if4, if5],
        constraints=[c1, c2, c3],
    )
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == (c1.rhs - 0.1) * -1
    assert len(constraints[0][0]) == len(c1.features) - 1
    assert len(constraints[0][1]) == len(c1.coefficients) - 1
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1, 2]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1, -1, -1]).to(**tkwargs))
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 2
    assert constraints[0][2] == c2.rhs * -1
    assert len(constraints[0][0]) == len(c2.features)
    assert len(constraints[0][1]) == len(c2.coefficients)
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1.0, -1.0]).to(**tkwargs))
    assert constraints[1][2] == (c3.rhs - 0.5 * 0.1) * -1
    assert len(constraints[1][0]) == len(c3.features) - 1
    assert len(constraints[1][1]) == len(c3.coefficients) - 1
    assert torch.allclose(constraints[1][0], torch.tensor([0, 1]))
    assert torch.allclose(constraints[1][1], torch.tensor([-1.0, -1.0]).to(**tkwargs))


def test_get_linear_constraints_unit_scaled():
    inputs = [
        ContinuousInput(
            key="base_polymer",
            bounds=(0.3, 0.7),
        ),
        ContinuousInput(
            key="glas_fibre",
            bounds=(0.1, 0.7),
        ),
        ContinuousInput(
            key="additive",
            bounds=(0.1, 0.6),
        ),
        ContinuousInput(
            key="temperature",
            bounds=(30, 700),
        ),
    ]
    constraints = [
        LinearEqualityConstraint(
            coefficients=[1.0, 1.0, 1.0],
            features=["base_polymer", "glas_fibre", "additive"],
            rhs=1.0,
        )
    ]
    domain = Domain(inputs=inputs, constraints=constraints)

    constraints = get_linear_constraints(
        domain, LinearEqualityConstraint, unit_scaled=True
    )
    assert len(constraints) == 1
    assert len(constraints[0][0]) == 3
    assert len(constraints[0][1]) == 3
    assert constraints[0][2] == 0.5 * -1
    assert torch.allclose(
        constraints[0][1], torch.tensor([0.4, 0.6, 0.5]).to(**tkwargs) * -1
    )
    assert torch.allclose(constraints[0][0], torch.tensor([1, 2, 0]))


of1 = ContinuousOutput(key="of1", objective=MaximizeObjective(w=1.0))
of2 = ContinuousOutput(
    key="of2",
    objective=MaximizeSigmoidObjective(
        w=1.0,
        tp=0,
        steepness=2,
    ),
)
of3 = ContinuousOutput(
    key="of3",
    objective=TargetObjective(w=1.0, tolerance=2, target_value=5, steepness=4),
)


@pytest.mark.parametrize(
    "outputs",
    [
        Outputs(features=[of1, of2, of3]),
        Outputs(features=[of2, of1, of3]),
    ],
)
def test_get_output_constraints(outputs):
    constraints, etas = get_output_constraints(outputs=outputs)
    assert len(constraints) == len(etas)
    assert np.allclose(etas, [0.5, 0.25, 0.25])


def test_get_nchoosek_constraints():
    domain = Domain(
        inputs=Inputs(
            features=[ContinuousInput(key=f"if{i+1}", bounds=(0, 1)) for i in range(8)]
        ),
        constraints=Constraints(
            constraints=[
                NChooseKConstraint(
                    features=[f"if{i+3}" for i in range(6)],
                    min_count=2,
                    max_count=5,
                    none_also_valid=False,
                )
            ]
        ),
    )
    constraints = get_nchoosek_constraints(domain=domain)
    assert len(constraints) == 2
    # wrong samples
    samples = domain.inputs.sample(5)
    # check max count not fulfilled
    assert torch.all(constraints[0](torch.from_numpy(samples.values).to(**tkwargs)) < 0)
    # check max count fulfilled
    samples.if8 = 0
    assert torch.all(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)) >= 0
    )

    # check min count fulfilled
    samples = domain.inputs.sample(5)
    assert torch.all(
        constraints[1](torch.from_numpy(samples.values).to(**tkwargs)) >= 0
    )
    samples[[f"if{i+4}" for i in range(5)]] = 0.0
    assert torch.all(constraints[1](torch.from_numpy(samples.values).to(**tkwargs)) < 0)
    domain = Domain(
        inputs=Inputs(
            features=[ContinuousInput(key=f"if{i+1}", bounds=(0, 1)) for i in range(8)]
        ),
        constraints=Constraints(
            constraints=[
                NChooseKConstraint(
                    features=[f"if{i+3}" for i in range(6)],
                    min_count=3,
                    max_count=6,
                    none_also_valid=False,
                )
            ]
        ),
    )
    constraints = get_nchoosek_constraints(domain=domain)
    assert len(constraints) == 1
    samples = domain.inputs.sample(5)
    assert torch.all(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)) >= 0
    )

    domain = Domain(
        inputs=Inputs(
            features=[ContinuousInput(key=f"if{i+1}", bounds=(0, 1)) for i in range(8)]
        ),
        constraints=Constraints(
            constraints=[
                NChooseKConstraint(
                    features=[f"if{i+3}" for i in range(6)],
                    min_count=0,
                    max_count=2,
                    none_also_valid=False,
                )
            ]
        ),
    )
    constraints = get_nchoosek_constraints(domain=domain)
    assert len(constraints) == 1
    samples = domain.inputs.sample(5)
    assert torch.all(constraints[0](torch.from_numpy(samples.values).to(**tkwargs)) < 0)
    # test with two max nchoosek constraints
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=0,
                max_count=1,
                none_also_valid=False,
            ),
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=0,
                max_count=2,
                none_also_valid=False,
            ),
        ],
    )
    samples = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to(**tkwargs)
    constraints = get_nchoosek_constraints(domain=domain)
    assert torch.allclose(
        constraints[0](samples), torch.tensor([0.0, -1.0, -2.0]).to(**tkwargs)
    )
    assert torch.allclose(
        constraints[1](samples), torch.tensor([1.0, 0.0, -1.0]).to(**tkwargs)
    )
    # test with two min nchoosek constraints
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=1,
                max_count=3,
                none_also_valid=False,
            ),
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=2,
                max_count=3,
                none_also_valid=False,
            ),
        ],
    )
    samples = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to(**tkwargs)
    constraints = get_nchoosek_constraints(domain=domain)
    assert torch.allclose(
        constraints[0](samples), torch.tensor([0.0, 1.0, 2.0]).to(**tkwargs)
    )
    assert torch.allclose(
        constraints[1](samples), torch.tensor([-1.0, 0.0, 1.0]).to(**tkwargs)
    )
    # test with min/max and max constraint
    # test with two min nchoosek constraints
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[0, 1]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=1,
                max_count=2,
                none_also_valid=False,
            ),
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=0,
                max_count=2,
                none_also_valid=False,
            ),
        ],
    )
    samples = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to(**tkwargs)
    constraints = get_nchoosek_constraints(domain=domain)
    assert torch.allclose(
        constraints[0](samples), torch.tensor([1.0, 0.0, -1.0]).to(**tkwargs)
    )
    assert torch.allclose(
        constraints[1](samples), torch.tensor([0.0, 1.0, 2.0]).to(**tkwargs)
    )
    assert torch.allclose(
        constraints[2](samples), torch.tensor([1.0, 0.0, -1.0]).to(**tkwargs)
    )


def test_get_product_constraints():
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[5, 100]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            ProductInequalityConstraint(
                features=["x2", "x3"],
                exponents=[1, 1],
                rhs=80,
            ),
            ProductInequalityConstraint(
                features=["x2", "x3"],
                exponents=[1, 1],
                rhs=-20,
                sign=-1,
            ),
            ProductInequalityConstraint(
                features=["x1", "x2", "x3"],
                exponents=[2, -1, 0.5],
                rhs=0,
                sign=-1,
            ),
        ],
    )
    constraints = get_product_constraints(domain=domain)
    assert len(constraints) == 3

    samples = torch.tensor([[0.1, 0.5, 90], [0.2, 0.9, 100], [0.3, 0.1, 100]]).to(
        **tkwargs
    )
    results = torch.tensor([35.0, -10.0, 70.0]).to(**tkwargs)
    assert torch.allclose(constraints[0](samples), results)
    for i in range(3):
        assert torch.allclose(constraints[0](samples[i]), results[i])

    results = torch.tensor([25.0, 70, -10]).to(**tkwargs)
    assert torch.allclose(constraints[1](samples), results)
    for i in range(3):
        assert torch.allclose(constraints[1](samples[i]), results[i])

    results = torch.tensor([0.18973666, 0.44444444444, 9.0]).to(**tkwargs)
    assert torch.allclose(constraints[2](samples), results)
    for i in range(3):
        assert torch.allclose(constraints[2](samples[i]), results[i])


def test_get_nonlinear_constraints():
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=[0, 1]),
            ContinuousInput(key="x2", bounds=[0, 1]),
            ContinuousInput(key="x3", bounds=[5, 100]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            ProductInequalityConstraint(
                features=["x2", "x3"],
                exponents=[1, 1],
                rhs=80,
            ),
            NChooseKConstraint(
                features=["x1", "x2"],
                min_count=0,
                max_count=1,
                none_also_valid=False,
            ),
        ],
    )
    assert len(get_nonlinear_constraints(domain=domain)) == 2


def test_get_multiobjective_objective():
    samples = (torch.rand(30, 4, requires_grad=True) * 5).to(**tkwargs)
    samples2 = (torch.rand(30, 512, 4, requires_grad=True) * 5).to(**tkwargs)
    a_samples = samples.detach().numpy()
    obj1 = MaximizeObjective()
    obj2 = MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5)
    obj3 = MinimizeObjective()
    obj4 = CloseToTargetObjective(target_value=2.5, exponent=1)
    outputs = Outputs(
        features=[
            ContinuousOutput(
                key="alpha",
                objective=obj1,
            ),
            ContinuousOutput(
                key="beta",
                objective=obj2,
            ),
            ContinuousOutput(
                key="gamma",
                objective=obj3,
            ),
            ContinuousOutput(
                key="omega",
                objective=obj4,
            ),
        ]
    )
    objective = get_multiobjective_objective(outputs=outputs)
    generic_objective = GenericMCObjective(objective=objective)
    # check the shape
    objective_forward = generic_objective.forward(samples2)
    assert objective_forward.shape == torch.Size((30, 512, 3))
    objective_forward = generic_objective.forward(samples)
    assert objective_forward.shape == torch.Size((30, 3))
    # check what is in
    # calc with numpy
    reward1 = obj1(a_samples[:, 0])
    reward3 = obj3(a_samples[:, 2])
    reward4 = obj4(a_samples[:, 3])
    assert np.allclose(objective_forward[..., 0].detach().numpy(), reward1)
    assert np.allclose(objective_forward[..., 1].detach().numpy(), reward3)
    assert np.allclose(objective_forward[..., 2].detach().numpy(), reward4)


@pytest.mark.parametrize("sequential", [True, False])
def test_get_initial_conditions_generator(sequential: bool):
    inputs = Inputs(
        features=[
            ContinuousInput(key="a", bounds=(0, 1)),
            CategoricalDescriptorInput(
                key="b",
                categories=["alpha", "beta", "gamma"],
                descriptors=["omega"],
                values=[[0], [1], [3]],
            ),
        ]
    )
    domain = Domain(inputs=inputs)
    strategy = strategies.map(RandomStrategy(domain=domain))
    # test with one hot encoding
    generator = get_initial_conditions_generator(
        strategy=strategy,
        transform_specs={"b": CategoricalEncodingEnum.ONE_HOT},
        ask_options={},
        sequential=sequential,
    )
    initial_conditions = generator(n=3, q=2, seed=42)
    assert initial_conditions.shape == torch.Size((3, 2, 4))
    # test with descriptor encoding
    generator = get_initial_conditions_generator(
        strategy=strategy,
        transform_specs={"b": CategoricalEncodingEnum.DESCRIPTOR},
        ask_options={},
        sequential=sequential,
    )
    initial_conditions = generator(n=3, q=2, seed=42)
    assert initial_conditions.shape == torch.Size((3, 2, 2))


@pytest.mark.parametrize(
    "objective",
    [
        (MaximizeSigmoidObjective(w=1, tp=15, steepness=0.5)),
        (MinimizeSigmoidObjective(w=1, tp=15, steepness=0.5)),
        (TargetObjective(w=1, target_value=15, steepness=2, tolerance=5)),
    ],
)
def test_constrained_objective2botorch(objective):
    cs, etas, _ = constrained_objective2botorch(idx=0, objective=objective)

    x = torch.from_numpy(np.linspace(0, 30, 500)).unsqueeze(-1).to(**tkwargs)

    result = (
        compute_smoothed_feasibility_indicator(
            constraints=cs,
            samples=x,
            eta=torch.tensor(etas).to(**tkwargs),
            log=False,
            fat=False,
        )
        .numpy()
        .ravel()
    )

    assert np.allclose(objective.__call__(np.linspace(0, 30, 500)), result)


def test_constrained_objective():
    desirability = [True, False, False]
    obj1 = ConstrainedCategoricalObjective(
        categories=["c1", "c2", "c3"], desirability=desirability
    )
    cs, etas, _ = constrained_objective2botorch(idx=0, objective=obj1)

    x = torch.zeros((50, 3))
    x[:, 0] = torch.arange(50) / 50
    true_y = (x * torch.tensor(desirability)).sum(-1)
    transformed_y = torch.log(1 / torch.clamp(true_y, 1e-8, 1 - 1e-8) - 1)

    assert len(cs) == 1
    assert etas[0] == 1.0

    y_hat = cs[0](x)
    assert np.allclose(y_hat.numpy(), transformed_y.numpy())
    assert (
        np.linalg.norm(
            np.exp(-np.log(np.exp(y_hat.numpy()) + 1)) - true_y.numpy(), ord=np.inf
        )
        <= 1e-8
    )
    result = (
        compute_smoothed_feasibility_indicator(
            constraints=cs,
            samples=x,
            eta=torch.tensor(etas).to(**tkwargs),
            log=False,
            fat=False,
        )
        .numpy()
        .ravel()
    )
    assert np.allclose(true_y.numpy(), result)
