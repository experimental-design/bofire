import importlib.util

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.doe import DOptimalityCriterion
from bofire.strategies.doe.design import (
    check_fixed_experiments,
    check_partially_and_fully_fixed_experiments,
    check_partially_fixed_experiments,
    find_local_max_ipopt,
    get_n_experiments,
)
from bofire.strategies.doe.objective import get_objective_function
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
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(4)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    dim_input = len(domain.inputs.get_keys())

    num_exp = (
        len(get_formula_from_string(model_type="linear", inputs=domain.inputs))
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )

    design = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=num_exp,
        ),
    )
    assert design.shape == (num_exp, dim_input)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_nchoosek():
    # Design for a problem with an n-choose-k constraint
    inputs = [
        ContinuousInput(
            key=f"x{i + 1}",
            bounds=(0, 1),
        )
        for i in range(4)
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=[f"x{i + 1}" for i in range(4)],
                min_count=0,
                max_count=3,
                none_also_valid=True,
            ),
        ],
    )

    D = len(domain.inputs)

    N = (
        len(get_formula_from_string(model_type="linear", inputs=domain.inputs))
        - n_zero_eigvals(domain=domain, model_type="linear")
        + 3
    )
    print(N)

    A = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=N,
        ),
    )
    assert A.shape == (N, D)


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_mixture():
    # Design for a problem with a mixture constraint
    inputs = [
        ContinuousInput(
            key=f"x{i + 1}",
            bounds=(0, 1),
        )
        for i in range(4)
    ]
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=[f"x{i + 1}" for i in range(4)],
                coefficients=[1, 1, 1, 1],
                rhs=1,
            ),
        ],
    )

    D = len(domain.inputs)

    N = len(get_formula_from_string(inputs=domain.inputs, model_type="linear")) + 3
    A = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=N,
        ),
    )
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
                features=[f"x{i + 1}" for i in range(3)],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            NChooseKConstraint(
                features=[f"x{i + 1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )

    N = (
        len(get_formula_from_string(model_type="fully-quadratic", inputs=domain.inputs))
        - n_zero_eigvals(domain=domain, model_type="fully-quadratic")
        + 3
    )
    # with pytest.warns(ValueError):
    A = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=N,
        ),
        ipopt_options={"max_iter": 100},
    )
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
                features=[f"x{i + 1}" for i in range(3)],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[5, 4],
                rhs=3.9,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[-20, 5],
                rhs=-3,
            ),
        ],
    )
    np.random.seed(1)
    A = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=12,
        ),
    )
    opt = np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]])
    for row in A.to_numpy():
        assert any(np.allclose(row, o, atol=1e-2) for o in opt)
    for o in opt[:-1]:
        assert any(np.allclose(o, row, atol=1e-2) for row in A.to_numpy())


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_find_local_max_ipopt_batch_constraint():
    # define problem with batch constraints
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[InterpointEqualityConstraint(features=["x1"], multiplicity=3)],
    )

    result = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=30,
        ),
        ipopt_options={"max_iter": 100},
    )

    x1 = np.round(np.array(result["x1"].values), 6)

    assert 0 in x1 and 1 in x1
    for i in range(10):
        assert x1[3 * i] == x1[3 * i + 1] and x1[3 * i] == x1[3 * i + 2]


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
                features=[f"x{i + 1}" for i in range(3)],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[5, 4],
                rhs=3.9,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[-20, 5],
                rhs=-3,
            ),
        ],
    )
    # np.random.seed(1)
    # fixed_experiments = pd.DataFrame([[0.3, 0.5, 0.2]], columns=["x1", "x2", "x3"])
    # A = find_local_max_ipopt(
    #     domain,
    #     "linear",
    #     n_experiments=12,
    #     fixed_experiments=fixed_experiments,
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
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=12,
            ),
            fixed_experiments=pd.DataFrame(
                np.ones(shape=(12, 3)),
                columns=["x1", "x2", "x3"],
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
                features=[f"x{i + 1}" for i in range(3)],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            NChooseKConstraint(
                features=[f"x{i + 1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )

    # with pytest.warns(ValueError):
    np.random.seed(1)

    num_exp = (
        len(get_formula_from_string(model_type="fully-quadratic", inputs=domain.inputs))
        - n_zero_eigvals(domain=domain, model_type="fully-quadratic")
        + 3
    )

    A = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="fully-quadratic"),
            domain=domain,
            n_experiments=num_exp,
        ),
        ipopt_options={"max_iter": 100},
        fixed_experiments=pd.DataFrame(
            [[1, 0, 0], [0, 1, 0]],
            columns=["x1", "x2", "x3"],
        ),
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
                features=[f"x{i + 1}" for i in range(3)],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            NChooseKConstraint(
                features=[f"x{i + 1}" for i in range(3)],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            ),
        ],
    )
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0], [0, 1, 0]]),
        columns=domain.inputs.get_keys(),
    )
    check_fixed_experiments(domain, 3, fixed_experiments)

    # define problem: not enough experiments
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0], [0, 1, 0]]),
        columns=domain.inputs.get_keys(),
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
            ),
        ],
    )

    num_exp = (
        len(get_formula_from_string(model_type="fully-quadratic", inputs=domain.inputs))
        - n_zero_eigvals(domain=domain, model_type="fully-quadratic")
        + 3
    )

    result = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=num_exp,
        ),
        ipopt_options={"max_iter": 100},
    )

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
    assert (
        get_n_experiments(get_formula_from_string("linear", inputs=domain.inputs)) == 7
    )

    # explicit formula
    assert (
        get_n_experiments(
            get_formula_from_string(
                "x1 + x2 + x3 + x1:x2 + {x2**2}", inputs=domain.inputs
            ),
        )
        == 9
    )

    # user provided n_experiment
    with pytest.warns(UserWarning):
        assert (
            get_n_experiments(
                get_formula_from_string("linear", inputs=domain.inputs), 4
            )
            == 4
        )


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_fixed_experiments_checker():
    domain = Domain.from_lists(
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
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, 10, -10],
                rhs=15,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 2, -2],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, -3, 3],
                rhs=5,
            ),
            # Case 2: a and c are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, -10, -10],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 2, 2],
                rhs=7,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, -3, -3],
                rhs=2,
            ),
            # Case 3: c and b are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, 0, -10],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 0, 2],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, 0, 3],
                rhs=5,
            ),
        ],
    )
    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
        columns=domain.inputs.get_keys(),
    )
    partially_fixed_experiments = pd.DataFrame(
        np.array([[1, None, None, None], [0, 1, 0, 0]]),
        columns=domain.inputs.get_keys(),
    )
    # all fine
    check_partially_and_fully_fixed_experiments(
        domain,
        10,
        fixed_experiments,
        partially_fixed_experiments,
    )

    # all fine
    check_partially_and_fully_fixed_experiments(
        domain,
        4,
        fixed_experiments,
        partially_fixed_experiments,
    )

    # partially fixed will be cut of
    with pytest.warns(UserWarning) as record:
        check_partially_and_fully_fixed_experiments(
            domain,
            3,
            fixed_experiments,
            partially_fixed_experiments,
        )
        assert len(record) == 1
        assert record[0].message.args[0] == (
            "The number of fixed experiments and partially fixed experiments exceeds the amount "
            "of the overall count of experiments. Partially fixed experiments may be cut off"
        )

    with pytest.warns(UserWarning) as record:
        check_partially_fixed_experiments(domain, 1, partially_fixed_experiments)
        assert len(record) == 1
        assert record[0].message.args[0] == (
            "The number of partially fixed experiments exceeds the amount "
            "of the overall count of experiments. Partially fixed experiments may be cut off"
        )

    # to few experiments
    with pytest.raises(ValueError) as e:
        check_partially_and_fully_fixed_experiments(
            domain,
            2,
            fixed_experiments,
            partially_fixed_experiments,
        )
        assert e == ValueError(
            "For starting the optimization the total number of experiments must be larger that the number of fixed experiments.",
        )

    with pytest.raises(ValueError) as e:
        check_fixed_experiments(domain, 2, fixed_experiments)
        assert e == ValueError(
            "For starting the optimization the total number of experiments must be larger that the number of fixed experiments.",
        )


def test_partially_fixed_experiments():
    pytest.importorskip("docutils")
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
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, 10, -10],
                rhs=15,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 2, -2],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, -3, 3],
                rhs=5,
            ),
            # Case 2: a and c are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, -10, -10],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 2, 2],
                rhs=7,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, -3, -3],
                rhs=2,
            ),
            # Case 3: c and b are active
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 1, 0, -10],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, 0.2, 0, 2],
                rhs=5,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "a1", "a2"],
                coefficients=[1, -1, 0, 3],
                rhs=5,
            ),
        ],
    )

    def get_domain_error(feature):
        return ValueError(f"no col for input feature `{feature}`")

    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0, 0], [0, 1, 0.7, 1]]),
        columns=domain.inputs.get_keys(),
    )

    doe = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=3,
        ),
        fixed_experiments=fixed_experiments,
    ).reset_index(drop=True)

    assert doe.shape == (3, 4)
    assert np.allclose(doe.iloc[[0, 1]]["x1"], fixed_experiments["x1"])
    assert np.allclose(doe.iloc[[0, 1]]["x2"], fixed_experiments["x2"])
    assert np.allclose(doe.iloc[[0, 1]]["a1"], fixed_experiments["a1"])
    assert np.allclose(doe.iloc[[0, 1]]["a2"], fixed_experiments["a2"])

    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0], [0, 1, 0.7]]),
        columns=["x1", "x2", "a1"],
    )

    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=2,
            ),
            fixed_experiments=fixed_experiments,
        )
        assert e == get_domain_error("a2")

    partially_fixed_experiments = pd.DataFrame(
        np.array([[1.0, None, None], [0.0, None, None]]),
        columns=["x1", "x2", "a1"],
    )

    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=2,
            ),
            partially_fixed_experiments=partially_fixed_experiments,
        )
        assert e == get_domain_error("a2")

    fixed_experiments = pd.DataFrame(
        np.array([[1, 0, 0, 0, 1], [0, 1, 0.7, 1, 2]]),
        columns=domain.inputs.get_keys() + ["c0"],
    )

    doe = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=3,
        ),
        fixed_experiments=fixed_experiments,
    ).reset_index(drop=True)

    assert doe.shape == (3, 4)
    assert np.allclose(doe.iloc[[0, 1]]["x1"], fixed_experiments["x1"])
    assert np.allclose(doe.iloc[[0, 1]]["x2"], fixed_experiments["x2"])
    assert np.allclose(doe.iloc[[0, 1]]["a1"], fixed_experiments["a1"])
    assert np.allclose(doe.iloc[[0, 1]]["a2"], fixed_experiments["a2"])

    partially_fixed_experiments = pd.DataFrame(
        np.array([[1.0, None, None, None, 1.0], [0.0, None, None, None, 2.0]]),
        columns=["x1", "x2", "a1", "a2", "c0"],
    )
    doe = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=3,
        ),
        partially_fixed_experiments=partially_fixed_experiments,
    ).reset_index(drop=True)

    assert doe.shape == (3, 4)
    assert np.allclose(
        doe.iloc[[0, 1]]["x1"],
        partially_fixed_experiments["x1"].astype(float),
    )

    doe = find_local_max_ipopt(
        domain,
        objective_function=get_objective_function(
            criterion=DOptimalityCriterion(formula="linear"),
            domain=domain,
            n_experiments=4,
        ),
        fixed_experiments=fixed_experiments,
        partially_fixed_experiments=partially_fixed_experiments,
    ).reset_index(drop=True)

    assert doe.shape == (4, 4)
    assert np.allclose(doe.iloc[[0, 1]]["x1"], fixed_experiments["x1"])
    assert np.allclose(doe.iloc[[0, 1]]["x2"], fixed_experiments["x2"])
    assert np.allclose(doe.iloc[[0, 1]]["a1"], fixed_experiments["a1"])
    assert np.allclose(doe.iloc[[0, 1]]["a2"], fixed_experiments["a2"])
    assert np.allclose(
        doe.iloc[[2, 3]]["x1"],
        partially_fixed_experiments["x1"].astype(float),
    )

    too_few_experiments_error = ValueError(
        "For starting the optimization the total number of experiments must be larger that the number of fixed experiments.",
    )
    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=1,
            ),
            fixed_experiments=fixed_experiments,
            partially_fixed_experiments=partially_fixed_experiments,
        )
        assert e == too_few_experiments_error
    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=2,
            ),
            fixed_experiments=fixed_experiments,
            partially_fixed_experiments=partially_fixed_experiments,
        )
        assert e == too_few_experiments_error

    _fixed_experiments = fixed_experiments.drop(columns=["x1"])
    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=3,
            ),
            fixed_experiments=_fixed_experiments,
            partially_fixed_experiments=partially_fixed_experiments,
        )
        assert e == get_domain_error("x1")

    _partially_fixed_experiments = partially_fixed_experiments.drop(columns=["x1"])
    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=3,
            ),
            fixed_experiments=fixed_experiments,
            partially_fixed_experiments=_partially_fixed_experiments,
        )
        assert e == ValueError(
            "Domain contains inputs that are not part of partially fixed experiments. Every input must be present as a column.",
        )

    with pytest.raises(ValueError) as e:
        doe = find_local_max_ipopt(
            domain,
            objective_function=get_objective_function(
                criterion=DOptimalityCriterion(formula="linear"),
                domain=domain,
                n_experiments=3,
            ),
            fixed_experiments=_fixed_experiments,
            partially_fixed_experiments=_partially_fixed_experiments,
        )
        assert e == ValueError(
            "Domain contains inputs that are not part of partially fixed experiments. Every input must be present as a column.",
        )
