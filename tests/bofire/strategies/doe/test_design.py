import importlib.util

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.strategies.doe.design import (
    check_fixed_experiments,
    check_partially_and_fully_fixed_experiments,
    find_local_max_ipopt,
    get_n_experiments,
)
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals

CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


@pytest.mark.skipif(CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_raise_error_if_cyipopt_not_available():
    pytest.raises(ImportError)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_no_constraint():
    # Design for a problem with an n-choose-k constraint
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(4)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    dim_input = len(domain.inputs.get_keys())

    num_exp = (
        len(get_formula_from_string(model_type="linear", domain=domain))
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )

    design = find_local_max_ipopt(domain, "linear")
    assert design.shape == (num_exp, dim_input)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_nchoosek():
    # Design for a problem with an n-choose-k constraint
    inputs = [
        ContinuousInput(
            key=f"x{i+1}",
            bounds=(0, 1),
        )
        for i in range(4)
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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
        len(get_formula_from_string(model_type="linear", domain=domain))
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )
    print(N)

    A = find_local_max_ipopt(domain, "linear")
    assert A.shape == (N, D)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_mixture():
    # Design for a problem with a mixture constraint
    inputs = [
        ContinuousInput(
            key=f"x{i+1}",
            bounds=(0, 1),
        )
        for i in range(4)
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i+1}" for i in range(4)], coefficients=[1, 1, 1, 1], rhs=1
            )
        ],
    )

    D = len(domain.inputs)

    N = len(get_formula_from_string(domain=domain, model_type="linear")) + 3
    A = find_local_max_ipopt(domain, "linear")
    assert A.shape == (N, D)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_mixed_results():
    inputs = [
        ContinuousInput(
            key=f"x{1}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{2}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{3}",
            bounds=(0, 1),
        ),
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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

    # with pytest.warns(ValueError):
    A = find_local_max_ipopt(domain, "fully-quadratic", ipopt_options={"maxiter": 100})
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any(np.allclose(row, o, atol=1e-2) for o in opt)
    for o in opt:
        assert any(np.allclose(o, row, atol=1e-2) for row in A.to_numpy())


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_results():
    # define problem: no NChooseK constraints
    inputs = [
        ContinuousInput(
            key=f"x{1}",
            bounds=(0, 1),
        ),
        ContinuousInput(key=f"x{2}", bounds=(0.1, 1)),
        ContinuousInput(key=f"x{3}", bounds=(0, 0.6)),
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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
        assert any(np.allclose(row, o, atol=1e-2) for o in opt)
    for o in opt[:-1]:
        assert any(np.allclose(o, row, atol=1e-2) for row in A.to_numpy())


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
    # TODO fix this test. Currently it gets stuck in a local minimum 50% of the time
    # working if fixed experiments is tested below. Do I need this test?
    inputs = [
        ContinuousInput(
            key=f"x{1}",
            bounds=(0, 1),
        ),
        ContinuousInput(key=f"x{2}", bounds=(0.1, 1)),
        ContinuousInput(key=f"x{3}", bounds=(0, 0.6)),
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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
    # np.random.seed(1)
    # fixed_experiments = pd.DataFrame([[0.3, 0.5, 0.2]], columns=["x1", "x2", "x3"])
    # A = find_local_max_ipopt(
    #     domain,
    #     "linear",
    #     n_experiments=12,
    #     fixed_experiments=fixed_experiments,  # type: ignore
    # )
    # opt = np.array(
    #     [
    #         [0.2, 0.2, 0.6],
    #         [0.3, 0.6, 0.1],
    #         [0.7, 0.1, 0.2],
    #         [0.3, 0.1, 0.6],
    #         [0.3, 0.5, 0.2],
    #     ]
    # )
    # for row in A.to_numpy():
    #     assert any([np.allclose(row, o, atol=1e-2) for o in opt])
    # for o in opt[:-1]:
    #     assert any([np.allclose(o, row, atol=1e-2) for row in A.to_numpy()])
    # assert np.allclose(A.to_numpy()[0, :], np.array([0.3, 0.5, 0.2]))

    # define domain: no NChooseK constraints, invalid proposal
    np.random.seed(1)
    with pytest.raises(ValueError):
        find_local_max_ipopt(
            domain,
            "linear",
            n_experiments=12,
            fixed_experiments=pd.DataFrame(
                np.ones(shape=(12, 3)), columns=["x1", "x2", "x3"]
            ),
        )

    # define domain: with NChooseK constraints, 2 fixed_experiments
    inputs = [
        ContinuousInput(
            key=f"x{1}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{2}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{3}",
            bounds=(0, 1),
        ),
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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

    # with pytest.warns(ValueError):
    np.random.seed(1)
    A = find_local_max_ipopt(
        domain,
        "fully-quadratic",
        ipopt_options={"maxiter": 100},
        fixed_experiments=pd.DataFrame([[1, 0, 0], [0, 1, 0]], columns=["x1", "x2", "x3"]),  # type: ignore
    )
    opt = np.eye(3)
    for row in A.to_numpy():
        assert any(np.allclose(row, o, atol=1e-2) for o in opt)
    for o in opt:
        assert any(np.allclose(o, row, atol=1e-2) for row in A.to_numpy())
    assert np.allclose(A.to_numpy()[:2, :], opt[:2, :])


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_check_fixed_experiments():
    # define problem: everything fine
    inputs = [
        ContinuousInput(
            key=f"x{1}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{2}",
            bounds=(0, 1),
        ),
        ContinuousInput(
            key=f"x{3}",
            bounds=(0, 1),
        ),
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
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
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0], [0, 1, 0]]), columns=domain.inputs.get_keys()
    )
    check_fixed_experiments(domain, 3, fixed_experiments)

    # define problem: not enough experiments
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0], [0, 1, 0]]), columns=domain.inputs.get_keys()
    )
    with pytest.raises(ValueError):
        check_fixed_experiments(domain, 2, fixed_experiments)

    # # define problem: invalid shape
    # fixed_experiments = pd.DataFrame(
    #     np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), columns=domain.inputs.get_keys()
    # )
    # with pytest.raises(ValueError):
    #     check_fixed_experiments(domain, 3, fixed_experiments)


# @pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
# def test_check_constraints_and_domain_respected():
#     # problem with unfulfillable constraints
#     # formulation constraint
#     inputs = [
#         ContinuousInput(key=f"x{1}", lower_bound=0.5, upper_bound=1),
#         ContinuousInput(key=f"x{2}", lower_bound=0.5, upper_bound=1),
#         ContinuousInput(key=f"x{3}", lower_bound=0.5, upper_bound=1),
#     ]
#     domain = Domain(
#         inputs=inputs,
#         outputs=[ContinuousOutput(key="y")],
#         constraints=[
#             LinearEqualityConstraint(
#                 features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
#             ),
#         ],
#     )

#     with warnings.catch_warnings():
#         # warnings.simplefilter("ignore")
#         try:
#             A = find_local_max_ipopt(domain=domain, model_type="linear")
#         except Exception as e:
#             assert (
#                 str(e)
#                 == "No feasible point found. Constraint polytope appears empty. Check your constraints."
#             )

#     # with pytest.warns(UserWarning, match="Please check your results"):
#     #    domain.validate_candidates(candidates=A, only_inputs=True)

#     # everything ok
#     inputs = [
#         ContinuousInput(key=f"x{1}", bounds=(0, 1),),
#         ContinuousInput(key=f"x{2}", bounds=(0, 1),),
#         ContinuousInput(key=f"x{3}", bounds=(0, 1),),
#     ]
#     domain = Domain(
#         inputs=inputs,
#         outputs=[ContinuousOutput(key="y")],
#         constraints=[
#             LinearEqualityConstraint(
#                 features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
#             ),
#             NChooseKConstraint(
#                 features=[f"x{i+1}" for i in range(3)],
#                 min_count=0,
#                 max_count=1,
#                 none_also_valid=True,
#             ),
#         ],
#     )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         A = find_local_max_ipopt(domain=domain, model_type="linear")

#     with warnings.catch_warnings():
#         warnings.simplefilter("error")
#         domain.validate_candidates(candidates=A, only_inputs=True)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_nonlinear_constraint():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(-1, 1)),
            ContinuousInput(key="x2", bounds=(-1, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 - x3",
                features=["x1", "x2", "x3"],
                jacobian_expression="[2*x1,2*x2,-1]",
            )
        ],
    )

    result = find_local_max_ipopt(domain, "linear", ipopt_options={"maxiter": 100})

    assert np.allclose(domain.constraints(result), 0, atol=1e-6)


def test_get_n_experiments():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(-1, 1)),
            ContinuousInput(key="x2", bounds=(-1, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # keyword
    assert get_n_experiments(domain, "linear") == 7

    # explicit formula
    assert get_n_experiments(domain, "x1 + x2 + x3 + x1:x2 + {x2**2}") == 9

    # user provided n_experiment
    with pytest.warns(UserWarning):
        assert get_n_experiments(domain, "linear", 4) == 4


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_partially_fixed_experiments():
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 5)),
            ContinuousInput(key="x2", bounds=(0, 15)),
            ContinuousInput(key="a1", bounds=(0, 1)),
            ContinuousInput(key="a2", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            # Case 1: a and b are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 1, 10, -10], rhs=15
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 0.2, 2, -2], rhs=5
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, -1, -3, 3], rhs=5
            ),
            # Case 2: a and c are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 1, -10, -10], rhs=5
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 0.2, 2, 2], rhs=7
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, -1, -3, -3], rhs=2
            ),
            # Case 3: c and b are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 1, 0, -10], rhs=5
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, 0.2, 0, 2], rhs=5
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"], coefficients=[1, -1, 0, 3], rhs=5
            ),
        ],
    )
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), columns=domain.inputs.get_keys()
    )
    partially_fixed_experiments = pd.DataFrame(
        np.array([[1, None, None, None], [0, 1, 0, 0]]),
        columns=domain.inputs.get_keys(),
    )
    # all fine
    check_partially_and_fully_fixed_experiments(
        domain, 10, fixed_experiments, partially_fixed_experiments
    )

    # all fine
    check_partially_and_fully_fixed_experiments(
        domain, 4, fixed_experiments, partially_fixed_experiments
    )

    # partially fixed will be cut of
    with pytest.warns(UserWarning):
        check_partially_and_fully_fixed_experiments(
            domain, 3, fixed_experiments, partially_fixed_experiments
        )

    # to few experiments
    with pytest.raises(ValueError):
        check_partially_and_fully_fixed_experiments(
            domain, 2, fixed_experiments, partially_fixed_experiments
        )
