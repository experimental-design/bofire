import itertools
import random

import numpy as np
import pytest
import torch
from sklearn.utils._testing import assert_allclose

from bofire.domain.objectives import (
    CloseToTargetObjective,
    ConstantObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
)
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)


@pytest.mark.parametrize(
    "objective, desFunc",
    [
        (objective, desFunc)
        for objective in [MultiplicativeObjective, AdditiveObjective]
        for desFunc in [
            DeltaObjective(w=0.5, ref_point=1.0, scale=0.8),
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=0.5),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=5.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(
                target_value=5.0, exponent=1.0, tolerance=1e-3, w=0.5
            ),
            ConstantObjective(w=0.5, value=1.0),
        ]
    ],
)
def test_Objective_not_implemented(objective, desFunc):
    one_objective = objective(desFunc)
    x = torch.rand(20, 1)

    with pytest.raises(NotImplementedError):
        one_objective.reward(x, None)


@pytest.mark.parametrize(
    "desFunc",
    [
        (desFunc)
        for desFunc in [
            DeltaObjective(w=0.5, ref_point=1.0, scale=0.8),
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=0.5),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=5.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(
                target_value=5.0, exponent=1.0, tolerance=1e-3, w=0.5
            ),
            ConstantObjective(w=0.5, value=1.0),
        ]
    ],
)
def test_Objective_desirability_function(desFunc):
    samples = torch.rand(20, 1, requires_grad=True)
    a_samples = samples.detach().numpy()
    objective = MultiplicativeObjective(desFunc)
    print(desFunc)
    print(objective.reward(samples, desFunc)[0].detach().numpy())
    assert_allclose(
        objective.reward(samples, desFunc)[0].detach().numpy(),
        np.sign(desFunc(a_samples)) * np.abs(desFunc(a_samples)) ** 0.5,
        rtol=1e-06,
    )

    objective = AdditiveObjective(desFunc)
    assert_allclose(
        objective.reward(samples, desFunc)[0].detach().numpy(),
        desFunc(a_samples) * 0.5,
        rtol=1e-06,
    )


@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype)
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        )
    ],
)
def test_Objective_max_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype, requires_grad=True)
    desFunc = MaximizeObjective(w=0.5)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], samples**0.5)

    objective = AdditiveObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], samples * 0.5)


@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype)
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        )
    ],
)
def test_Objective_min_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype, requires_grad=True)
    desFunc = MinimizeObjective(w=0.5)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], -1.0 * samples**0.5)

    objective = AdditiveObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], -1.0 * samples * 0.5)


@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype)
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        )
    ],
)
def test_Objective_delta_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype, requires_grad=True)

    desFunc = DeltaObjective(w=0.5, ref_point=5.0, scale=0.8)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(
        objective.reward(samples, desFunc)[0], ((5 - samples) * 0.8) ** 0.5
    )

    objective = AdditiveObjective(desFunc)
    assert torch.equal(
        objective.reward(samples, desFunc)[0], ((5 - samples) * 0.8) * 0.5
    )


def test_MultiplicativeObjective_forward():
    (desFunc, desFunc2) = random.choices(
        [
            DeltaObjective(w=0.5, ref_point=1.0),
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=0.5),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=5.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(
                target_value=5.0, exponent=1.0, tolerance=1e-3, w=1.0
            ),
            ConstantObjective(w=0.5, value=1.0),
        ],
        k=2,
    )

    objective = MultiplicativeObjective([desFunc, desFunc2])

    samples = torch.rand(20, 2, requires_grad=True)
    reward, _ = objective.reward(samples[:, 0], desFunc)
    reward2, _ = objective.reward(samples[:, 1], desFunc2)

    exp_reward = reward.detach().numpy() * reward2.detach().numpy()

    forward_reward = objective.forward(samples)

    assert_allclose(exp_reward, forward_reward.detach().numpy(), rtol=1e-06)


def test_AdditiveObjective_forward():
    (desFunc, desFunc2) = random.choices(
        [
            DeltaObjective(w=0.5, ref_point=1.0),
            MaximizeObjective(w=0.5),
            MaximizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            MinimizeObjective(w=0.5),
            MinimizeSigmoidObjective(steepness=1.0, tp=1.0, w=0.5),
            TargetObjective(target_value=5.0, steepness=1.0, tolerance=1e-3, w=0.5),
            CloseToTargetObjective(
                target_value=5.0, exponent=1.0, tolerance=1e-3, w=1.0
            ),
            ConstantObjective(w=0.5, value=1.0),
        ],
        k=2,
    )

    objective = AdditiveObjective([desFunc, desFunc2])

    samples = torch.rand(20, 2, requires_grad=True)
    reward, _ = objective.reward(samples[:, 0], desFunc)
    reward2, _ = objective.reward(samples[:, 1], desFunc2)

    exp_reward = reward.detach().numpy() + reward2.detach().numpy()

    forward_reward = objective.forward(samples)

    assert_allclose(exp_reward, forward_reward.detach().numpy(), rtol=1e-06)


# TODO: test sigmoid behaviour
# @pytest.mark.parametrize(
#     "batch_shape, m, dtype, desFunc",
#     [
#         (batch_shape, m, dtype, desFunc)
#         for batch_shape, m, dtype, desFunc in itertools.product(
#         ([], [3]),
#         (2, 3),
#         (torch.float, torch.double),
#         (MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.), MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.))
#         )
#     ],
# )
# def test_MultiplicativeObjective_sigmoid(batch_shape, m, dtype, desFunc):
#     objective = MultiplicativeObjective([desFunc])

#     samples = torch.rand(*batch_shape, 20, m, dtype=dtype)
#     reward, _ = objective.reward(samples, desFunc)

# assert torch.equal(torch.topk(reward, 1, dim=0).indices, torch.topk(samples, 1, dim=0).indices)
# assert torch.equal(torch.topk(reward, 1, largest=False, dim=0).indices, torch.topk(samples, 1, largest=False, dim=0).indices)

# sort_samples, indices = torch.sort(samples, dim=- 1, descending=False)
# delta_middle_sample = sort_samples[9,...]-sort_samples[8,...]
# delta_middle_reward = reward[...,indices[9,...]]-reward[...,indices[8,...]]

# delta_high_sample = sort_samples[0,...]-sort_samples[1,...]
# delta_high_reward = reward[...,indices[0,...]]-reward[...,indices[1,...]]

# assert delta_high_reward<delta_middle_reward
# assert delta_high_reward<delta_high_sample
# assert_allclose(delta_middle_sample, delta_middle_reward)
