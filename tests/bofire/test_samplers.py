import pytest

from bofire.domain import Domain
from bofire.domain.constraints import Constraints, LinearInequalityConstraint
from bofire.domain.features import ContinuousInput, InputFeatures
from bofire.samplers import RejectionSampler

input_features = InputFeatures(
    features=[
        ContinuousInput(key=f"if{i}", lower_bound=0, upper_bound=1) for i in range(1, 4)
    ]
)
constraints = Constraints(
    constraints=[
        LinearInequalityConstraint(
            features=["if1", "if2", "if3"], coefficients=[1, 1, 1], rhs=1
        )
    ]
)


@pytest.mark.parametrize(
    "features, constraints, sampling_method, num_samples",
    [
        (input_features, constraints, sampling_method, num_samples)
        for sampling_method in ["SOBOL", "UNIFORM", "LHS"]
        for num_samples in [1, 2, 64, 128]
    ],
)
def test_rejection_sampler(features, constraints, sampling_method, num_samples):
    domain = Domain(
        input_features=features,
        constraints=constraints,
    )
    sampler = RejectionSampler(domain=domain, sampling_method=sampling_method)
    sampler(num_samples)


def test_rejection_sampler_not_converged():
    pass
