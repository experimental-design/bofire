from typing import List

import pytest
import pandas as pd
import numpy as np
import cvxopt
from scipy import sparse, linalg

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.strategies import api as data_models_strategies
from bofire.data_models.features import api as data_models_features
from bofire.strategies.predictives.utils import LinearProjection, get_problem_and_algorithm, GaMixedDomainHandler
from bofire.strategies.predictives.acqf_optimization import AcquisitionOptimizer
from bofire.utils.torch_tools import get_linear_constraints


@pytest.fixture
def domain_handler(optimizer_benchmark) -> GaMixedDomainHandler:
    """Fixture to provide a problem and algorithm for testing."""
    domain = optimizer_benchmark.get_adapted_domain()
    input_preprocessing_specs = optimizer_benchmark.get_strategy(
        optimizer=data_models_strategies.GeneticAlgorithmOptimizer(),  # dummy
    ).input_preprocessing_specs

    q = optimizer_benchmark.n_add

    return GaMixedDomainHandler(
        domain=domain,
        input_preprocessing_specs=input_preprocessing_specs,
        q=q,
    )


@pytest.fixture
def mock_pymoo_generation(optimizer_benchmark) -> List[dict]:
    """Fixture to provide a problem and algorithm for testing."""
    domain = optimizer_benchmark.get_adapted_domain()
    q = optimizer_benchmark.n_add

    N = 10  # mock generation size

    data = []
    for qi in range(q):
        # Sample N points from the domain for each generation
        sample = domain.inputs.sample(N)

        for input_ in domain.inputs.get():
            if isinstance(input_, data_models_features.DiscreteInput):
                map = {val: i for (i, val) in enumerate(input_.values)}
                sample[input_.key] = sample[input_.key].map(map)

        sample.columns = [f"{col}_q{qi}" for col in sample.columns]
        data.append(sample)

    data = pd.concat(data, axis=1)
    return data.to_dict(orient="records")


@pytest.fixture
def repair_instance(optimizer_benchmark, domain_handler) -> LinearProjection:
    """Fixture to provide a problem and algorithm for testing."""

    domain = optimizer_benchmark.get_adapted_domain()
    strategy: data_models_strategies.BotorchStrategy = optimizer_benchmark.get_strategy(
        optimizer=data_models_strategies.GeneticAlgorithmOptimizer(),  # dummy
    )

    input_preprocessing_specs = strategy.input_preprocessing_specs
    bounds_botorch_space = AcquisitionOptimizer.get_bounds(domain, input_preprocessing_specs)
    q = optimizer_benchmark.n_add

    # We handle linear equality constraint with a repair function
    repair = LinearProjection(
        domain=domain,
        d=bounds_botorch_space.shape[1],
        bounds=bounds_botorch_space,
        q=q,
        domain_handler=domain_handler,
    )

    return repair

class TestLinearProjection:


    def test_create_qp_problem(self, mock_pymoo_generation: List[dict], repair_instance: LinearProjection):

        n_gen = len(mock_pymoo_generation)
        n_add = repair_instance.q
        d = repair_instance.d
        domain = repair_instance.domain

        X = repair_instance.domain_handler.transform_mixed_to_2D(mock_pymoo_generation)

        matrices = repair_instance._create_qp_problem_input(X)

        P, q, G, h, A, b = (matrices.get(key) for key in ["P", "q", "G", "h", "A", "b"])

        # numpy conversion
        P, q, G, h, A, b = (np.array(cvxopt.matrix(x)) for x in [P, q, G, h, A, b])

        # check objective: x^T P x + q^T x
        assert (P == np.eye(n_gen * n_add * d)).all()
        assert (X.reshape(-1) == -q.reshape(-1)).all()

        # box-bounds (G/h matrices)
        G_bounds = G[:2* n_gen * n_add * d, :]
        assert G_bounds.shape == (2 * n_gen * n_add * d, n_gen * n_add * d)
        h_bounds = h[:2 * n_gen * n_add * d]
        assert len(h_bounds) == 2 * n_gen * n_add * d
        # assert structure of G
        assert (linalg.block_diag(*[np.vstack([np.eye(d), -np.eye(d)]) for _ in range(n_gen * n_add)]) == G_bounds).all()

        nck_constr = domain.constraints.get(includes=NChooseKConstraint).constraints

        for xi in range(n_gen * n_add):
            ub = h_bounds[(xi*2) * d : ((xi*2) + 1) * d]
            lb = -h_bounds[((xi*2) + 1) * d : ((xi*2) + 2) * d]

            if len(nck_constr) > 0:
                # check how NChooseK constraints manipulate the bounds
                for idx_constr in range(len(nck_constr)):
                    idx = repair_instance.n_choose_k_constr.idx[idx_constr]
                    lb, ub = lb[idx], ub[idx]
                    assert int((lb>0).sum()) >= nck_constr[idx_constr].min_count
                    assert int((ub==0).sum()) <= len(ub) - nck_constr[idx_constr].max_count

            else:
                assert (ub.reshape(-1) == repair_instance.bounds[1, :].numpy().reshape(-1)).all()
                assert (lb.reshape(-1) == repair_instance.bounds[0, :].numpy().reshape(-1)).all()

        lin_ineq = domain.constraints.get(LinearInequalityConstraint)
        n_constr = len(lin_ineq.constraints)
        if n_constr > 0:
            G_constr = G[2 * n_gen * n_add * d:, :]
            h_constr = h[2 * n_gen * n_add * d:]
            assert G_constr.shape[0] == n_constr * n_add * n_gen

        lin_eq = domain.constraints.get(LinearEqualityConstraint)
        n_constr = len(lin_eq.constraints)
        if n_constr > 0:
            assert A.shape[0] == n_constr * n_add * n_gen