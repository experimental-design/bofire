import numpy as np
import pytest
import torch

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
    MaximizeObjective,
    MaximizeSigmoidObjective,
    TargetObjective,
)
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_nchoosek_constraints,
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
    assert torch.allclose(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.ones(5).to(**tkwargs) * -1,
        rtol=1e-4,
        atol=1e-6,
    )
    # check max count fulfilled
    samples.if8 = 0
    assert torch.allclose(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.zeros(5).to(**tkwargs),
        rtol=1e-4,
        atol=1e-6,
    )

    # check min count fulfilled
    samples = domain.inputs.sample(5)
    assert torch.allclose(
        constraints[1](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.ones(5).to(**tkwargs) * 4,
        rtol=1e-4,
        atol=1e-6,
    )
    # check min count not fulfilled
    samples[[f"if{i+4}" for i in range(5)]] = 0.0
    assert torch.allclose(
        constraints[1](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.ones(5).to(**tkwargs) * -1,
        rtol=1e-4,
        atol=1e-6,
    )
    # check no creation of max_count constraint if max_count = n_features
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
    assert torch.allclose(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.ones(5).to(**tkwargs) * 3,
        rtol=1e-4,
        atol=1e-6,
    )
    # check no creation of min_count constraint if min_count = 0
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
    assert torch.allclose(
        constraints[0](torch.from_numpy(samples.values).to(**tkwargs)),
        torch.ones(5).to(**tkwargs) * -4,
        rtol=1e-4,
        atol=1e-6,
    )
