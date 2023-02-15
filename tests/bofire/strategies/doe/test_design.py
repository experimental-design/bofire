import warnings

import importlib.util
import numpy as np
import pytest

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.features import ContinuousInput, ContinuousOutput
from bofire.strategies.doe.design import (
    check_fixed_experiments,
    find_local_max_ipopt,
    get_objective,
    logD,
)
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals

CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


@pytest.mark.skipif(CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_raise_error_if_cyipopt_not_available():
    pytest.raises(ImportError)


def test_logD():
    design = np.ones(shape=(10, 5))
    design[0, 0] = 2

    log_d = logD(design)
    log_d_true = np.linalg.slogdet(design.T @ design + 1e-7 * np.eye(5))[1]

    assert np.allclose(log_d, log_d_true)


def test_get_objective():
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )
    objective = get_objective(domain=domain, model_type="linear")

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(objective(x), -np.log(4) - np.log(1e-7))


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_no_constraint():
    # Design for a problem with an n-choose-k constraint
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
    )
    dim_input = len(domain.inputs.get_keys())

    num_exp = (
        len(get_formula_from_string(model_type="linear", domain=domain).terms)
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )

    design = find_local_max_ipopt(domain, "linear")
    assert design.shape == (num_exp, dim_input)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_nchoosek():
    # Design for a problem with an n-choose-k constraint
    input_features = [
        ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1) for i in range(4)
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=[f"x{i+1}" for i in range(4)],
                min_count=0,
                max_count=3,
                none_also_valid=True,
            )
        ],
    )

    D = len(domain.inputs)

    N = (
        len(get_formula_from_string(model_type="linear", domain=domain).terms)
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )

    A = find_local_max_ipopt(domain, "linear")
    assert A.shape == (N, D)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_mixture():
    # Design for a problem with a mixture constraint
    input_features = [
        ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1) for i in range(4)
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(4)], coefficients=[1, 1, 1, 1], rhs=1
            )
        ],
    )

    D = len(domain.inputs)

    N = len(get_formula_from_string(domain=domain, model_type="linear").terms) + 3
    A = find_local_max_ipopt(domain, "linear")
    assert A.shape == (N, D)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_mixed_results():
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=1),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            NChooseKConstraint(
                features=[f"x{i+1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )

    with pytest.warns(UserWarning):
        A = find_local_max_ipopt(
            domain, "fully-quadratic", ipopt_options={"maxiter": 100}
        )
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_results():
    # define problem: no NChooseK constraints
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0.1, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=0.6),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[5, 4], rhs=3.9
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[-20, 5], rhs=-3
            ),
        ],
    )
    np.random.seed(1)
    A = find_local_max_ipopt(domain, "linear", n_experiments=12)
    opt = np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]])
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt[:-1]:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])


# def test_find_local_max_ipopt_sampling():
#     # define problem
#     problem = opti.Problem(
#         inputs=[opti.Continuous(f"x{i}", [0, 1]) for i in range(3)],
#         outputs=[opti.Continuous("y")],
#     )


#     # test sampling methods
#     find_local_max_ipopt(problem, "linear", sampling=OptiSampling)
#     find_local_max_ipopt(problem, "linear", sampling=CornerSampling)
#     find_local_max_ipopt(problem, "linear", sampling=ProbabilitySimplexSampling)
#     sampling = np.zeros(shape=(10, 3)).flatten()
#     find_local_max_ipopt(problem, "linear", n_experiments=10, sampling=sampling)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_fixed_experiments():
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0.1, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=0.6),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[5, 4], rhs=3.9
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[-20, 5], rhs=-3
            ),
        ],
    )
    np.random.seed(3)
    A = find_local_max_ipopt(
        domain,
        "linear",
        n_experiments=12,
        fixed_experiments=[[0.3, 0.5, 0.2]],  # type: ignore
    )
    print(A)
    opt = np.array(
        [
            [0.2, 0.2, 0.6],
            [0.3, 0.6, 0.1],
            [0.7, 0.1, 0.2],
            [0.3, 0.1, 0.6],
            [0.3, 0.5, 0.2],
        ]
    )
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt[:-1]:
        print(o)
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])
    assert np.allclose(A.to_numpy()[0, :], np.array([0.3, 0.5, 0.2]))

    # define domain: no NChooseK constraints, invalid proposal
    with pytest.raises(ValueError):
        find_local_max_ipopt(
            domain,
            "linear",
            n_experiments=12,
            fixed_experiments=np.ones(shape=(12, 3)),
        )

    # define domain: with NChooseK constraints, 2 fixed_experiments
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=1),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            NChooseKConstraint(
                features=[f"x{i+1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )

    with pytest.warns(UserWarning):
        A = find_local_max_ipopt(
            domain,
            "fully-quadratic",
            ipopt_options={"maxiter": 100},
            fixed_experiments=[[1, 0, 0], [0, 1, 0]],  # type: ignore
        )
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    for o in opt:
        assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])
    assert np.allclose(A.to_numpy()[:2, :], opt[:2, :])


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_check_fixed_experiments():
    # define problem: everything fine
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=1),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            NChooseKConstraint(
                features=[f"x{i+1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )
    fixed_experiments = np.array([[1, 0, 0], [0, 1, 0]])
    check_fixed_experiments(domain, 3, fixed_experiments)

    # define problem: not enough experiments
    fixed_experiments = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        check_fixed_experiments(domain, 2, fixed_experiments)

    # define problem: invalid shape
    fixed_experiments = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    with pytest.raises(ValueError):
        check_fixed_experiments(domain, 3, fixed_experiments)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_check_constraints_and_domain_respected():
    # problem with unfulfillable constraints
    # formulation constraint
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0.5, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0.5, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0.5, upper_bound=1),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
        ],
    )

    with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        try:
            A = find_local_max_ipopt(domain=domain, model_type="linear")
        except Exception as e:
            assert (
                str(e)
                == "No feasible point found. Constraint polytope appears empty. Check your constraints."
            )

    # with pytest.warns(UserWarning, match="Please check your results"):
    #    domain.validate_candidates(candidates=A, only_inputs=True)

    # everything ok
    input_features = [
        ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{2}", lower_bound=0, upper_bound=1),
        ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=1),
    ]
    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
            ),
            NChooseKConstraint(
                features=[f"x{i+1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = find_local_max_ipopt(domain=domain, model_type="linear")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        domain.validate_candidates(candidates=A, only_inputs=True)
