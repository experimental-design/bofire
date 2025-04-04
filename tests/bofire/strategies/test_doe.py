import warnings

import numpy as np
import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.doe import DOptimalityCriterion
from bofire.strategies.api import DoEStrategy


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
        optimization_strategy="partially-random",
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
        optimization_strategy="partially-random",
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=n_experiments)

    assert candidates.shape == (10, 9)


def test_partially_fixed_experiments():
    continuous_var = [
        ContinuousInput(key=f"continuous_var_{i}", bounds=(100, 230)) for i in range(2)
    ]

    all_constraints = [
        NChooseKConstraint(
            features=[var.key for var in continuous_var],
            min_count=1,
            max_count=2,
            none_also_valid=True,
        ),
    ]
    all_inputs = [
        CategoricalInput(key="animal", categories=["dog", "whale", "cat"]),
        CategoricalInput(key="plant", categories=["tulip", "sunflower"]),
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
        optimization_strategy="relaxed",
        verbose=True,
    )
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(
        pd.DataFrame(
            [
                [150, 100, 0.3, 0.2, None, None],
                [0, 100, 0.3, 0.2, None, "tulip"],
                [0, 100, None, 0.2, "dog", None],
                [0, 100, 0.3, 0.2, "cat", "tulip"],
                [None, 100, 0.3, None, None, None],
            ],
            columns=[
                "continuous_var_0",
                "continuous_var_1",
                "a_discrete",
                "b_discrete",
                "animal",
                "plant",
            ],
        ),
    )

    only_partially_fixed = pd.DataFrame(
        [
            [150, 100, 0.3, 0.2, None, None],
            [0, 100, 0.3, 0.2, None, "tulip"],
            [0, 100, None, 0.2, "dog", None],
            [None, 100, 0.3, None, None, None],
        ],
        columns=[
            "continuous_var_0",
            "continuous_var_1",
            "a_discrete",
            "b_discrete",
            "animal",
            "plant",
        ],
    )

    candidates = strategy.ask(candidate_count=n_experiments)
    only_partially_fixed = only_partially_fixed.mask(
        only_partially_fixed.isnull(),
        candidates[:4],
    )
    test_df = pd.DataFrame(np.ones((4, 6)))
    test_df = test_df.where(candidates[:4] == only_partially_fixed, 0)
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


def test_categorical_doe_iterative():
    quantity_a = [
        ContinuousInput(key=f"quantity_a_{i}", bounds=(20, 100)) for i in range(2)
    ]
    all_inputs = [
        ContinuousInput(key="independent", bounds=(3, 10)),
    ]
    all_inputs.extend(quantity_a)

    all_constraints = [
        NChooseKConstraint(
            features=[var.key for var in quantity_a],
            min_count=1,
            max_count=1,
            none_also_valid=False,
        ),
    ]

    n_experiments = 5
    domain = Domain.from_lists(
        inputs=all_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=all_constraints,
    )

    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=DOptimalityCriterion(formula="linear"),
        optimization_strategy="iterative",
    )
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(
        candidate_count=n_experiments,
        raise_validation_error=False,
    )

    assert candidates.shape == (5, 3)


def test_functional_constraint():
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
        return np.array([A, B, T, W, W_T]).T @ (df_raw_materials["sc"].values)

    # Calculate the volume content of the formulation
    def calc_volume_content(A, B, T, W, W_T):
        A = A.detach() if not isinstance(A, pd.Series) else A
        B = B.detach() if not isinstance(B, pd.Series) else B
        T = T.detach() if not isinstance(T, pd.Series) else T
        W = W.detach() if not isinstance(W, pd.Series) else W
        W_T = W_T.detach() if not isinstance(W_T, pd.Series) else W_T
        volume_solid = (
            A * raw_materials_data["A"][0] / raw_materials_data["A"][1]
            + B * raw_materials_data["B"][0] / raw_materials_data["B"][1]
        )
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
        ipopt_options={"max_iter": 500},
    )
    strategy = DoEStrategy(data_model=data_model)
    doe = strategy.ask(candidate_count=n_experiments, raise_validation_error=False)
    doe["SC"] = calc_solid_content(*[doe[col] for col in ["A", "B", "T", "W", "W_T"]])
    doe["VC"] = calc_volume_content(*[doe[col] for col in ["A", "B", "T", "W", "W_T"]])
    doe["T_calc"] = 0.0182 - 0.03704 * doe["VC"]
    doe["T_conc"] = doe["T"] / (doe["T"] + doe["W_T"])

    assert np.allclose(doe["T_conc"], 0.03)
    assert all((doe["VC"] > 0.299) & (doe["VC"] < 0.45))


if __name__ == "__main__":
    test_functional_constraint()
