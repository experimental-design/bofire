import random

import numpy as np
import pytest
import torch
from botorch.acquisition.objective import GenericMCObjective

from bofire.domain.constraint import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.constraints import Constraints
from bofire.domain.domain import Domain
from bofire.domain.feature import CategoricalInput, ContinuousInput, ContinuousOutput
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.domain.objective import (
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
)
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_multiplicative_botorch_objective,
    get_nchoosek_constraints,
    get_objective_callable,
    get_output_constraints,
    tkwargs,
)

if1 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if1",
)
if2 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if2",
)
if3 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if3",
)
if4 = ContinuousInput(
    lower_bound=0.1,
    upper_bound=0.1,
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
        DeltaObjective(w=0.5, ref_point=1.0, scale=0.8),
        MaximizeObjective(w=0.5),
        MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
        MinimizeObjective(w=0.5),
        MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
        TargetObjective(target_value=2.0, steepness=1.0, tolerance=1e-3, w=0.5),
        CloseToTargetObjective(target_value=2.0, exponent=1.0, tolerance=1e-3, w=0.5),
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


def test_get_objective_callable_not_implemented():
    with pytest.raises(NotImplementedError):
        get_objective_callable(idx=1, objective=ConstantObjective(w=0.5, value=1.0))


def test_get_multiplicative_botorch_objective():
    (obj1, obj2) = random.choices(
        [
            DeltaObjective(w=0.5, ref_point=10.0),
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=1),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=2.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(
                target_value=2.0, exponent=1.0, tolerance=1e-3, w=1.0
            ),
        ],
        k=2,
    )
    output_features = OutputFeatures(
        features=[
            ContinuousOutput(key="alpha", objective=obj1),
            ContinuousOutput(key="beta", objective=obj2),
        ]
    )
    objective = get_multiplicative_botorch_objective(output_features)
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


def test_get_linear_constraints():
    domain = Domain(input_features=[if1, if2])
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 0

    domain = Domain(input_features=[if1, if2, if3], constraints=[c2])
    constraints = get_linear_constraints(domain, LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = get_linear_constraints(domain, LinearInequalityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c2.rhs * -1
    assert torch.allclose(constraints[0][0], torch.tensor([0, 1]))
    assert torch.allclose(constraints[0][1], torch.tensor([-1.0, -1.0]).to(**tkwargs))

    domain = Domain(
        input_features=[if1, if2, if3, if4],
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
        input_features=[if1, if2, if3, if4, if5],
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
    input_features = [
        ContinuousInput(key="base_polymer", lower_bound=0.3, upper_bound=0.7),
        ContinuousInput(key="glas_fibre", lower_bound=0.1, upper_bound=0.7),
        ContinuousInput(key="additive", lower_bound=0.1, upper_bound=0.6),
        ContinuousInput(key="temperature", lower_bound=30.0, upper_bound=700.0),
    ]
    constraints = [
        LinearEqualityConstraint(
            coefficients=[1.0, 1.0, 1.0],
            features=["base_polymer", "glas_fibre", "additive"],
            rhs=1.0,
        )
    ]
    domain = Domain(input_features=input_features, constraints=constraints)

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
    "output_features",
    [
        OutputFeatures(features=[of1, of2, of3]),
        OutputFeatures(features=[of2, of1, of3]),
    ],
)
def test_get_output_constraints(output_features):
    constraints, etas = get_output_constraints(output_features=output_features)
    assert len(constraints) == len(etas)
    assert np.allclose(etas, [0.5, 0.25, 0.25])


def test_get_nchoosek_constraints():
    domain = Domain(
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"if{i+1}", lower_bound=0, upper_bound=1)
                for i in range(8)
            ]
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
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"if{i+1}", lower_bound=0, upper_bound=1)
                for i in range(8)
            ]
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
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"if{i+1}", lower_bound=0, upper_bound=1)
                for i in range(8)
            ]
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
