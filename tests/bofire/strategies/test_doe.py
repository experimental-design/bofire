import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
)
from bofire.data_models.constraints.constraint import ConstraintNotFulfilledError
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.doe import (
    DOptimalityCriterion,
    SpaceFillingCriterion,
)
from bofire.strategies.api import DoEStrategy
from bofire.strategies.doe.utils import get_formula_from_string


# from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)


pytest.importorskip("cyipopt")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

inputs = [
    ContinuousInput(
        key=f"x{1}",
        bounds=(0.0, 1.0),
    ),
    ContinuousInput(
        key=f"x{2}",
        bounds=(0.1, 1.0),
    ),
    ContinuousInput(
        key=f"x{3}",
        bounds=(0.0, 0.6),
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
        LinearInequalityConstraint(features=["x1", "x2"], coefficients=[5, 4], rhs=3.9),
        LinearInequalityConstraint(
            features=["x1", "x2"],
            coefficients=[-20, 5],
            rhs=-3,
        ),
    ],
)


def test_doe_strategy_init():
    data_model = data_models.DoEStrategy(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    assert strategy is not None


def test_doe_strategy_ask():
    data_model = data_models.DoEStrategy(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_doe_strategy_ask_with_candidates():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_nchoosek_implemented():
    nchoosek_constraint = NChooseKConstraint(
        features=[f"x{i + 1}" for i in range(3)],
        min_count=0,
        max_count=2,
        none_also_valid=True,
    )
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[nchoosek_constraint],
    )
    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_formulas_implemented():
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
    )
    expected_num_candidates = {
        "linear": 7,  # 1+a+b+c+3
        "linear-and-quadratic": 10,  # 1+a+b+c+a**2+b**2+c**2+3
        "linear-and-interactions": 10,  # 1+a+b+c+ab+ac+bc+3
        "fully-quadratic": 13,  # 1+a+b+c+a**2+b**2+c**2+ab+ac+bc+3
    }

    for formula, num_candidates in expected_num_candidates.items():
        data_model = data_models.DoEStrategy(
            domain=domain, criterion=DOptimalityCriterion(formula=formula)
        )
        strategy = DoEStrategy(data_model=data_model)
        candidates = strategy.ask(strategy.get_required_number_of_experiments())
        assert candidates.shape == (num_candidates, 3)


def test_doe_strategy_correctness():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)

    np.random.seed(1)
    candidates_expected = np.array(
        [[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]],
    )
    for row in candidates.to_numpy():
        assert any(np.allclose(row, o, atol=1e-2) for o in candidates_expected)
    for o in candidates_expected[:-1]:
        assert any(np.allclose(o, row, atol=1e-2) for row in candidates.to_numpy())


def test_doe_strategy_amount_of_candidates():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)

    np.random.seed(1)
    num_candidates_expected = 12
    assert len(candidates) == num_candidates_expected


def test_categorical_discrete_doe():
    quantity_a = [
        ContinuousInput(key=f"quantity_a_{i}", bounds=(0, 100)) for i in range(3)
    ]
    quantity_b = [
        ContinuousInput(key=f"quantity_b_{i}", bounds=(0, 15)) for i in range(3)
    ]
    all_inputs = [
        CategoricalInput(key="animals", categories=["Whale", "Turtle", "Sloth"]),
        DiscreteInput(key="discrete", values=[0.1, 0.2, 0.3, 1.6, 2]),
        ContinuousInput(key="independent", bounds=(3, 10)),
    ]
    all_inputs.extend(quantity_a)
    all_inputs.extend(quantity_b)

    all_constraints = [
        NChooseKConstraint(
            features=[var.key for var in quantity_a],
            min_count=0,
            max_count=1,
            none_also_valid=True,
        ),
        NChooseKConstraint(
            features=[var.key for var in quantity_b],
            min_count=0,
            max_count=2,
            none_also_valid=True,
        ),
        LinearEqualityConstraint(
            features=[var.key for var in quantity_b],
            coefficients=[1 for var in quantity_b],
            rhs=15,
        ),
    ]

    n_experiments = 10
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        scip_params={"parallel/maxnthreads": 1, "numerics/feastol": 1e-8},
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=n_experiments)

    assert candidates.shape == (10, 9)


def test_partially_fixed_experiments():
    continuous_var = [
        ContinuousInput(key=f"continuous_var_{i}", bounds=(0, 230)) for i in range(2)
    ]

    all_constraints = [
        NChooseKConstraint(
            features=[var.key for var in continuous_var],
            min_count=0,
            max_count=2,
            none_also_valid=True,
        ),
    ]
    all_inputs = [
        DiscreteInput(key="a_discrete", values=[0.1, 0.2, 0.3, 1.6, 2]),
        DiscreteInput(key="b_discrete", values=[0.1, 0.2, 0.3, 1.6, 2]),
    ]
    n_experiments = 10

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        return_fixed_candidates=True,
        scip_params={"parallel/maxnthreads": 1, "numerics/feastol": 1e-8},
    )
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(
        pd.DataFrame(
            [
                [150, 100, 0.3, 0.2],
                [
                    0,
                    100,
                    0.3,
                    0.2,
                ],
                [0, 100, None, 0.2],
                [0, 100, 0.3, 0.2],
                [None, 100, 0.3, None],
            ],
            columns=[
                "continuous_var_0",
                "continuous_var_1",
                "a_discrete",
                "b_discrete",
            ],
        ),
    )

    only_partially_fixed = pd.DataFrame(
        [
            [150, 100, 0.3, 0.2],
            [0, 100, 0.3, 0.2],
            [0, 100, None, 0.2],
            [None, 100, 0.3, None],
        ],
        columns=[
            "continuous_var_0",
            "continuous_var_1",
            "a_discrete",
            "b_discrete",
        ],
    )

    candidates = strategy.ask(candidate_count=n_experiments)
    only_partially_fixed = only_partially_fixed.mask(
        only_partially_fixed.isnull(),
        candidates[:4],
    )
    test_df = pd.DataFrame(np.ones((4, 6)))
    test_df = test_df.where(
        candidates[:4][
            [
                "continuous_var_0",
                "continuous_var_1",
                "a_discrete",
                "b_discrete",
            ]
        ]
        == only_partially_fixed[
            [
                "continuous_var_0",
                "continuous_var_1",
                "a_discrete",
                "b_discrete",
            ]
        ],
        0,
    )
    assert test_df.sum().sum() == 0


def test_scaled_doe():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{1}",
                bounds=(0.0, 1.0),
            ),
            ContinuousInput(
                key=f"x{2}",
                bounds=(0.0, 1.0),
            ),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[],
    )
    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear", transform_range=(-1, 1)),
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=6).to_numpy()
    expected_candidates = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    for c in candidates:
        assert np.any([np.allclose(c, e) for e in expected_candidates])


@pytest.mark.xfail(reason="This test is failing due to a bad initial point.")
def test_functional_constraint():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sampling = [
        [
            0.36531936,
            0.774256,
            0.0457612,
            0.10232313,
            0.66522677,
        ],
        [
            0.39283984,
            0.30091256,
            0.3751583,
            0.72181405,
            0.37838284,
        ],
        [
            0.38254448,
            0.35407898,
            0.47484388,
            0.337307,
            0.76760944,
        ],
        [0.31534748, 0.51893979, 0.42009135, 0.30568969, 0.85966],
    ]

    inputs = [
        ContinuousInput(key="A", bounds=(0.2, 0.4)),
        ContinuousInput(key="B", bounds=(0, 0.8)),
        ContinuousInput(key="T", bounds=(0, 1)),
        ContinuousInput(key="W_T", bounds=(0, 1)),
        ContinuousInput(key="W", bounds=(0, 1)),
    ]

    outputs = [ContinuousOutput(key="y")]

    # Aggregate the solids content as well as the density of the materials in a dictionary
    # First col: solids content, second col= density
    raw_materials_data = {
        "A": [0.4, 2],
        "B": [1, 1.5],
        "T": [1, 1],
        "W": [0, 1],
        "W_T": [0, 1],
    }

    df_raw_materials = pd.DataFrame(raw_materials_data, index=["sc", "density"]).T

    # Mixture constraint: All components should sum up to 1
    constraint1 = LinearEqualityConstraint(
        features=["A", "B", "T", "W", "W_T"], coefficients=[1, 1, 1, 1, 1], rhs=1.0
    )
    # Set the lower bound of the volume content to 0.3.
    constraint2 = LinearInequalityConstraint(
        features=["A", "B", "W_T", "W"], coefficients=[0.04, -0.467, 0.3, 0.3], rhs=0
    )

    # Set the upper bound of the volume content to 0.45.
    constraint3 = LinearInequalityConstraint(
        features=["A", "B", "W_T", "W"],
        coefficients=[-0.16, 0.367, -0.45, -0.45],
        rhs=0,
    )

    # Calculate the solid content of the formulation
    def calc_solid_content(A, B, T, W, W_T):
        # Ensure same order as in the dictionary containing the material properties
        if isinstance(A, torch.Tensor):
            return torch.stack([A, B, T, W, W_T], 0).T @ torch.tensor(
                df_raw_materials["sc"].values
            )
        else:
            return np.array([A, B, T, W, W_T]).T @ df_raw_materials["sc"].values

    # Calculate the volume content of the formulation
    def calc_volume_content(A, B, T, W, W_T):
        volume_solid = (
            A * raw_materials_data["A"][0] / raw_materials_data["A"][1]
            + B * raw_materials_data["B"][0] / raw_materials_data["B"][1]
        )
        A = A
        B = B
        T = T
        W = W
        W_T = W_T
        volume_total = volume_solid + (1 - calc_solid_content(A, B, T, W, W_T) / 1)
        return volume_solid / volume_total

    constraint5 = NonlinearEqualityConstraint(
        expression=lambda A, B, T, W, W_T: T
        - 0.0182
        + 0.03704 * calc_volume_content(A, B, T, W, W_T),
    )

    # Set the thinner solution to 3 %.
    constraint4 = LinearEqualityConstraint(
        features=["T", "W_T"], coefficients=[0.97, -0.03], rhs=0
    )

    n_experiments = 4
    domain = Domain.from_lists(
        inputs=inputs,
        outputs=outputs,
        constraints=[
            constraint1,
            constraint2,
            constraint3,
            constraint4,
            constraint5,
        ],
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        ipopt_options={
            "max_iter": 5000,
            "derivative_test": "first-order",
            "print_level": 5,
        },
        sampling=sampling,
        use_cyipopt=True,
    )
    strategy = DoEStrategy(data_model=data_model)

    doe = strategy.ask(candidate_count=n_experiments, raise_validation_error=True)

    doe["SC"] = calc_solid_content(*[doe[col] for col in ["A", "B", "T", "W", "W_T"]])
    doe["VC"] = calc_volume_content(*[doe[col] for col in ["A", "B", "T", "W", "W_T"]])
    doe["T_calc"] = 0.0182 - 0.03704 * doe["VC"]
    doe["T_conc"] = doe["T"] / (doe["T"] + doe["W_T"])

    assert np.allclose(doe["T_conc"], 0.03)
    assert all((doe["VC"] > 0.299) & (doe["VC"] < 0.45))


def test_free_discrete_doe():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    all_inputs = [
        DiscreteInput(key="a_discrete", values=[-0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
        DiscreteInput(key="b_discrete", values=[-0.1, 0.0, 0.2, 0.3, 1.6], rtol=1e-3),
        DiscreteInput(key="c_discrete", values=[0.0, 5, 8, 10], rtol=1e-3),
    ]
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    n_exp = strategy.get_required_number_of_experiments()
    candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
    assert candidates.shape == (n_exp, 3)
    # check only lb and ub are values in candidates
    for col in all_inputs:
        assert np.all(
            [
                np.any([np.isclose(v, u) for v in [col.values[0], col.values[-1]]])
                for u in candidates[col.key]
            ]
        ), f"Column {col.key} contains values outside of the bounds."


def test_discrete_and_categorical_doe_w_constraints():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    continuous_var = [
        ContinuousInput(key=f"continuous_var_{i}", bounds=[0, 1]) for i in range(2)
    ]
    all_inputs = [
        DiscreteInput(key="a_discrete", values=[0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
        DiscreteInput(
            key="b_discrete", values=[-1.0, 0.0, 0.2, 0.3, 1.6, 10], rtol=1e-3
        ),
        DiscreteInput(key="c_discrete", values=[0.0, 5, 8, 10], rtol=1e-3),
        CategoricalInput(key="flatulent_butterfly", categories=["pff", "pf", "pffpff"]),
        CategoricalInput(key="farting_turtle", categories=["meep", "moop"]),
    ]
    all_constraints = [
        LinearInequalityConstraint(
            features=["a_discrete", f"continuous_var_{1}"],
            coefficients=[-1, -1],
            rhs=-1.9,
        ),
        # NChooseKConstraint(
        #     features=["b_discrete", f"continuous_var_{0}"],
        #     min_count=0,
        #     max_count=1,
        #     none_also_valid=True,
        # ),
    ]

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    n_exp = strategy.get_required_number_of_experiments()
    candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
    assert candidates.shape == (n_exp, 7)


def test_discrete_and_categorical_doe_w_constraints_num_of_experiments():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    continuous_var = [ContinuousInput(key="a", bounds=[0, 1]) for i in range(1)]
    all_inputs = [
        DiscreteInput(key="b", values=[0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
        CategoricalInput(key="c", categories=["meep", "moop"]),
    ]
    all_constraints = [
        LinearInequalityConstraint(
            features=["b", "a"],
            coefficients=[-1, -1],
            rhs=-1.9,
        ),
    ]

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    excepted_num_candidates = {
        "linear": 7,  # 1+a+b+c+3
        "linear-and-quadratic": 9,  # 1+a+b+(c==meep)+(1-(c==meep))+a**2+b**2+3
        "linear-and-interactions": 10,  # 1+a+b+c==meep+ab+a(c==meep)+b(c==meep)+3
        "fully-quadratic": 12,  # 1+a+b+c+a**2+b**2+ab+ac+bc+3
    }

    excepted_model_string = {
        "linear": "1 + a + b + aux_c_meep",
        "linear-and-quadratic": "1 + a + b + aux_c_meep + a ** 2 + b ** 2",
        "linear-and-interactions": "1 + a + b + aux_c_meep + a:aux_c_meep + a:b + aux_c_meep:b",
        "fully-quadratic": "1 + a + b + aux_c_meep + a ** 2 + b ** 2 + a:aux_c_meep + a:b + aux_c_meep:b",
    }

    for model_type in [
        "linear",
        "linear-and-quadratic",
        "linear-and-interactions",
        "fully-quadratic",
    ]:
        data_model = data_models.DoEStrategy(
            domain=domain,
            criterion=DOptimalityCriterion(formula=model_type),
            verbose=True,
            scip_params={"parallel/maxnthreads": 1},
        )

        formula = get_formula_from_string(model_type=model_type, inputs=domain.inputs)
        assert str(formula) == excepted_model_string[model_type]

        strategy = DoEStrategy(data_model=data_model)
        n_exp = strategy.get_required_number_of_experiments()
        assert (
            n_exp == excepted_num_candidates[model_type]
        ), f"Expected {excepted_num_candidates[model_type]} candidates for {model_type}, got {n_exp}"
        candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
        assert candidates.shape == (
            n_exp,
            3,
        ), f"Expected {n_exp} candidates, got {candidates.shape[0]}"

    continuous_var = [ContinuousInput(key="a", bounds=[0, 1]) for i in range(1)]
    all_inputs = [
        DiscreteInput(key="b", values=[0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
        CategoricalInput(key="c", categories=["meep", "moop", "moep"]),
    ]
    all_constraints = [
        LinearInequalityConstraint(
            features=["b", "a"],
            coefficients=[-1, -1],
            rhs=-1.9,
        ),
    ]

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    excepted_num_candidates = {
        "linear": 8,  # 1+a+b+c+3
        "linear-and-quadratic": 10,  # 1+a+b+(c==meep)+(c==moop)+a**2+b**2+3
        "linear-and-interactions": 13,  # 1+a+b+(c==meep)+(c==moop)+ab+a(c==meep)+a(c==moop)+b(c==meep)+b(c==moop)+3
        "fully-quadratic": 15,  # 1+a+b+(c==meep)+(c==moop)+ab+a(c==meep)+a(c==moop)+b(c==meep)+b(c==moop)+a**2+b**2+3
    }
    excepted_model_string = {
        "linear": "1 + a + b + aux_c_meep + aux_c_moop",
        "linear-and-quadratic": "1 + a + b + aux_c_meep + aux_c_moop + a ** 2 + b ** 2",
        "linear-and-interactions": "1 + a + b + aux_c_meep + aux_c_moop + a:aux_c_meep + a:aux_c_moop + a:b + aux_c_meep:b + aux_c_moop:b",
        "fully-quadratic": "1 + a + b + aux_c_meep + aux_c_moop + a ** 2 + b ** 2 + a:aux_c_meep + a:aux_c_moop + a:b + aux_c_meep:b + aux_c_moop:b",
    }

    for model_type in [
        "linear",
        "linear-and-quadratic",
        "linear-and-interactions",
        "fully-quadratic",
    ]:
        data_model = data_models.DoEStrategy(
            domain=domain,
            criterion=DOptimalityCriterion(formula=model_type),
            verbose=True,
            scip_params={"parallel/maxnthreads": 1},
        )

        formula = get_formula_from_string(model_type=model_type, inputs=domain.inputs)
        assert str(formula) == excepted_model_string[model_type]

        strategy = DoEStrategy(data_model=data_model)
        n_exp = strategy.get_required_number_of_experiments()
        assert (
            n_exp == excepted_num_candidates[model_type]
        ), f"Expected {excepted_num_candidates[model_type]} candidates, got {n_exp}"
        candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
        assert candidates.shape == (
            n_exp,
            3,
        ), f"Expected {n_exp} candidates, got {candidates.shape[0]}"


def test_compare_discrete_to_continuous_mapping_with_thresholding():
    continuous_var = [
        ContinuousInput(key=f"continuous_var_{i}", bounds=[0, 1]) for i in range(2)
    ]
    all_inputs = [
        ContinuousInput(key="a_discrete", bounds=[0.1, 2]),
        ContinuousInput(key="b_discrete", bounds=[0.1, 2]),
    ]
    all_constraints = [
        LinearInequalityConstraint(
            features=["a_discrete", f"continuous_var_{1}"],
            coefficients=[-1, -1],
            rhs=-1.9,
        ),
    ]

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=5, raise_validation_error=True)

    # Apply thresholding
    a_discrete_grid = [0.1, 0.2, 0.3, 1.6, 2]
    b_discrete_grid = [0.1, 0.2, 0.3, 1.6, 2]

    candidates["a_discrete"] = candidates["a_discrete"].apply(
        lambda x: min(a_discrete_grid, key=lambda y: abs(x - y))
    )
    candidates["b_discrete"] = candidates["b_discrete"].apply(
        lambda x: min(b_discrete_grid, key=lambda y: abs(x - y))
    )

    try:
        # validate the candidates
        domain.validate_candidates(
            candidates=candidates,
            only_inputs=True,
            raise_validation_error=True,
        )
    except ConstraintNotFulfilledError as e:
        assert isinstance(e, ConstraintNotFulfilledError)

    continuous_var = [
        ContinuousInput(key=f"continuous_var_{i}", bounds=[0, 1]) for i in range(2)
    ]
    all_inputs = [
        DiscreteInput(key="a_discrete", values=[0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
        DiscreteInput(key="b_discrete", values=[0.1, 0.2, 0.3, 1.6, 2], rtol=1e-3),
    ]
    all_constraints = [
        LinearInequalityConstraint(
            features=["a_discrete", f"continuous_var_{1}"],
            coefficients=[-1, -1],
            rhs=-1.9,
        ),
    ]

    all_inputs = all_inputs + continuous_var
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=5, raise_validation_error=True)
    assert candidates.shape == (5, 4)


def test_purely_categorical_doe():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    all_inputs = [
        CategoricalInput(key="gaseous_eel", categories=["blub", "bloop"]),
        CategoricalInput(key="flatulent_butterfly", categories=["pff", "pf", "pffpff"]),
        CategoricalInput(key="farting_turtle", categories=["meep", "moop"]),
    ]

    domain = Domain.from_lists(
        inputs=all_inputs, outputs=[ContinuousOutput(key="y")], constraints=[]
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    n_exp = strategy.get_required_number_of_experiments()
    candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
    assert candidates.shape == (n_exp, 3)


def test_continuous_categorical_doe():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    all_inputs = [
        CategoricalInput(key="gaseous_eel", categories=["blub", "bloop"]),
        CategoricalInput(key="flatulent_butterfly", categories=["pff", "pf", "pffpff"]),
        ContinuousInput(key="x0", bounds=(0, 1)),
        CategoricalInput(key="farting_turtle", categories=["meep", "moop"]),
    ]

    domain = Domain.from_lists(
        inputs=all_inputs, outputs=[ContinuousOutput(key="y")], constraints=[]
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    n_exp = strategy.get_required_number_of_experiments()
    candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
    assert candidates.shape == (n_exp, 4)


def one_cont_3_cat():
    np.random.seed(42)
    torch.manual_seed(42)
    # Test case: extra experiments
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="cat1", categories=["miau", "meow"]),
            CategoricalInput(key="cat2", categories=["oink", "oinki", "grunt"]),
            CategoricalInput(key="cat3", categories=["wuff", "wuffwuff", "ruff"]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=SpaceFillingCriterion(),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)
    n_successfull_runs = 0
    for i in range(10):
        candidates = strategy.ask(candidate_count=20, raise_validation_error=True)
        assert np.shape(candidates.to_numpy()) == (20, 4)
        n_successfull_runs = i
    assert n_successfull_runs == 9


def test_get_candidate_rank():
    """Test the get_candidate_rank method of DoEStrategy."""
    # Create a simple domain with 3 continuous inputs
    simple_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # Test 1: No candidates should return 0
    data_model = data_models.DoEStrategy(
        domain=simple_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    assert strategy.get_candidate_rank() == 0

    # Test 2: Full rank Fisher Information Matrix (4 candidates for linear model: intercept + 3 variables)
    candidates_full_rank = pd.DataFrame(
        {
            "x1": [1.0, 0.0, 0.0, 0.5],
            "x2": [0.0, 1.0, 0.0, 0.5],
            "x3": [0.0, 0.0, 1.0, 0.5],
        }
    )
    strategy.set_candidates(candidates_full_rank)
    rank = strategy.get_candidate_rank()
    assert rank == 4  # Intercept + 3 variables = 4 estimable parameters

    # Test 3: Rank-deficient Fisher Information Matrix (linearly dependent design points)
    candidates_rank_deficient = pd.DataFrame(
        {
            "x1": [
                1.0,
                1.0,
                0.0,
                0.0,
            ],  # Two pairs of identical rows creates linear dependence in design matrix
            "x2": [0.0, 0.0, 1.0, 1.0],
            "x3": [
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # Constant, provides no information beyond intercept
        }
    )
    strategy.set_candidates(candidates_rank_deficient)
    rank = strategy.get_candidate_rank()
    assert (
        rank == 2
    )  # Only 2 linearly independent design points: spans intercept + 1 direction

    # Test 4: Test with quadratic formula
    data_model_quad = data_models.DoEStrategy(
        domain=simple_domain, criterion=DOptimalityCriterion(formula="fully-quadratic")
    )
    strategy_quad = DoEStrategy(data_model_quad)
    strategy_quad.set_candidates(candidates_full_rank)
    rank_quad = strategy_quad.get_candidate_rank()
    # Fully quadratic has 10 terms (excluding intercept), with 4 candidates rank is 4
    assert rank_quad == 4

    # Test 5: SpaceFilling criterion should raise error
    data_model_space = data_models.DoEStrategy(
        domain=simple_domain, criterion=SpaceFillingCriterion()
    )
    strategy_space = DoEStrategy(data_model_space)
    strategy_space.set_candidates(candidates_full_rank)

    with pytest.raises(
        ValueError,
        match="get_candidate_rank\\(\\) only works with DoEOptimalityCriterion",
    ):
        strategy_space.get_candidate_rank()


def test_get_candidate_rank_categorical_discrete():
    """Test the get_candidate_rank method with categorical and discrete inputs."""
    # Create a domain with mixed input types
    mixed_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            DiscreteInput(key="x2", values=[0.1, 0.5, 1.0]),
            CategoricalInput(key="x3", categories=["A", "B", "C"]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # Test 1: No candidates should return 0
    data_model = data_models.DoEStrategy(
        domain=mixed_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    assert strategy.get_candidate_rank() == 0

    # Test 2: Mixed input candidates
    candidates_mixed = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, 0.2],
            "x2": [0.1, 0.5, 1.0, 0.5],  # Discrete values
            "x3": ["A", "B", "C", "A"],  # Categorical values
        }
    )
    strategy.set_candidates(candidates_mixed)
    rank = strategy.get_candidate_rank()
    # With proper encoding, all 4 candidates are linearly independent
    assert rank == 4

    # Test 3: Test with interactions formula for mixed types
    data_model_interactions = data_models.DoEStrategy(
        domain=mixed_domain,
        criterion=DOptimalityCriterion(formula="linear-and-interactions"),
    )
    strategy_interactions = DoEStrategy(data_model_interactions)
    strategy_interactions.set_candidates(candidates_mixed)
    rank_interactions = strategy_interactions.get_candidate_rank()
    # With interactions, rank is still 4 (limited by number of candidates)
    assert rank_interactions == 4

    # Test 4: Rank-deficient case with repeated categorical/discrete values
    candidates_repeated = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 0.5, 0.5],  # Repeated continuous values
            "x2": [0.1, 0.1, 0.5, 0.5],  # Repeated discrete values
            "x3": ["A", "A", "B", "B"],  # Repeated categorical values
        }
    )
    strategy.set_candidates(candidates_repeated)
    rank_repeated = strategy.get_candidate_rank()
    assert (
        rank_repeated == 2
    )  # Only 2 unique design points: intercept + 1 independent direction

    # Test 5: Single categorical level (should reduce rank)
    candidates_single_cat = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, 0.2],
            "x2": [0.1, 0.5, 1.0, 0.1],
            "x3": ["A", "A", "A", "A"],  # All same category
        }
    )
    strategy.set_candidates(candidates_single_cat)
    rank_single = strategy.get_candidate_rank()
    # Intercept + x1 + x2 (x3 categorical doesn't vary, contributes no information)
    assert rank_single == 3


def test_get_additional_experiments_needed():
    """Test the get_additional_experiments_needed method."""
    simple_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    data_model = data_models.DoEStrategy(
        domain=simple_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)

    # Test 1: No candidates - should return full required number
    required = strategy.get_required_number_of_experiments()
    assert required is not None
    assert strategy.get_additional_experiments_needed() == required

    # Test 2: Partial rank - should return the difference
    candidates_partial = pd.DataFrame(
        {"x1": [0.0, 1.0], "x2": [0.0, 0.0], "x3": [0.0, 0.0]}
    )
    strategy.set_candidates(candidates_partial)
    rank = strategy.get_candidate_rank()
    assert strategy.get_additional_experiments_needed() == required - rank
    assert rank < required
    additional_needed = strategy.get_additional_experiments_needed()
    assert additional_needed is not None
    assert additional_needed > 0

    # Test 3: Generate a full DoE from scratch and verify it has full rank
    # Using the same domain and criterion as Test 1 and 2
    full_doe = strategy.ask(candidate_count=required)

    # Create a fresh strategy instance and set the full DoE as candidates
    strategy_fresh = DoEStrategy(data_model=data_model)
    strategy_fresh.set_candidates(full_doe)
    rank_full_doe = strategy_fresh.get_candidate_rank()

    # A properly generated D-optimal DoE should have rank ~ model terms
    formula = get_formula_from_string(
        strategy._data_model.criterion.formula, inputs=strategy.domain.inputs
    )
    assert rank_full_doe == len(
        formula
    ), f"Expected DoE to have rank {len(formula)}, but got {rank_full_doe}"

    # Test 4: SpaceFilling criterion should return None
    data_model_sf = data_models.DoEStrategy(
        domain=simple_domain, criterion=SpaceFillingCriterion()
    )
    strategy_sf = DoEStrategy(data_model=data_model_sf)
    assert strategy_sf.get_additional_experiments_needed() is None


def test_custom_formula_with_categorical_and_discrete():
    """Test DoE strategy with custom formula containing categorical interactions."""
    from formulaic import Formula

    from bofire.data_models.domain.api import Inputs

    np.random.seed(42)
    torch.manual_seed(42)

    # Create a domain with categorical, continuous, and discrete variables
    inputs = Inputs(
        features=[
            CategoricalInput(
                key="color",
                categories=["red", "blue", "green"],
            ),
            CategoricalInput(
                key="material",
                categories=["plastic", "metal"],
            ),
            ContinuousInput(
                key="temperature",
                bounds=(20.0, 100.0),
            ),
            DiscreteInput(
                key="pressure",
                values=[1.0, 2.0, 3.0, 5.0, 10.0],
            ),
        ]
    )

    domain = Domain(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
    )

    # Define a custom formula with interactions among categorical variables
    custom_formula = Formula(
        "color + material + temperature + pressure + color:material"
    )

    # Create DoE strategy with the custom formula
    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula=str(custom_formula)),
        verbose=True,
        scip_params={"parallel/maxnthreads": 1},
    )
    strategy = DoEStrategy(data_model=data_model)

    # Get required number of experiments
    n_exp = strategy.get_required_number_of_experiments()
    assert n_exp is not None
    # Formula has: 1 (intercept) + 2 (color) + 1 (material) + 1 (temp) + 1 (pressure) + 2 (color:material) = 8 terms
    assert n_exp == 8

    # Generate candidates
    candidates = strategy.ask(candidate_count=n_exp, raise_validation_error=True)
    assert candidates.shape == (n_exp, 4)

    # Verify all categorical values are valid
    assert all(candidates["color"].isin(["red", "blue", "green"]))
    assert all(candidates["material"].isin(["plastic", "metal"]))

    # Verify continuous and discrete values are within bounds
    assert all(
        (candidates["temperature"] >= 20.0) & (candidates["temperature"] <= 100.0)
    )
    assert all(candidates["pressure"].isin([1.0, 2.0, 3.0, 5.0, 10.0]))


if __name__ == "__main__":
    test_custom_formula_with_categorical_and_discrete()
