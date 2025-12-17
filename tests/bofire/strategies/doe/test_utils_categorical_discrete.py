import pandas as pd

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.doe.utils_categorical_discrete import (
    create_continuous_domain,
    encode_candidates_to_relaxed_domain,
    filter_out_categorical_and_categorical_auxilliary_vars,
    filter_out_discrete_auxilliary_vars,
    project_candidates_into_domain,
)


def test_domain_relaxation():
    domain = Domain(
        inputs=Inputs(
            features=[
                DiscreteInput(key="x1", values=[-0.3, 5]),
                DiscreteInput(key="x2", values=[0.7, 10]),
                ContinuousInput(key="x3", bounds=[10, 11]),
                ContinuousInput(key="x4", bounds=[5, 11]),
                CategoricalInput(key="x5", categories=["a", "b", "c"]),
            ]
        ),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(
                    features=["x1", "x2"],
                    coefficients=[-1, -1],
                    rhs=-10,
                ),
                LinearEqualityConstraint(
                    features=["x3", "x4"],
                    coefficients=[1, 1],
                    rhs=15,
                ),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y1")]),
    )
    (
        relaxed_domain,
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux,
        aux_vars_for_discrete,
        mapped_aux_categorical_inputs,
        mapped_continous_inputs,
    ) = create_continuous_domain(domain=domain)

    assert len(relaxed_domain.inputs.get([DiscreteInput])) == 0
    assert len(relaxed_domain.inputs.get([ContinuousInput])) == 11
    assert len(relaxed_domain.constraints.get([LinearInequalityConstraint])) == 1
    assert len(relaxed_domain.constraints.get([LinearEqualityConstraint])) == 6
    assert len(relaxed_domain.constraints.get([NChooseKConstraint])) == 0

    assert len(mappings_categorical_var_key_to_aux_var_key_state_pairs) == 1
    assert len(mapping_discrete_input_to_discrete_aux) == 2
    assert len(aux_vars_for_discrete) == 4

    for x in ["x1", "x2"]:
        assert len(mapping_discrete_input_to_discrete_aux[x]) == 2

    assert len(mapped_continous_inputs) == 2

    assert "aux_x1___neg__0__decpt__3" in mapping_discrete_input_to_discrete_aux["x1"]
    assert "aux_x1_5__decpt__0" in mapping_discrete_input_to_discrete_aux["x1"]

    assert "aux_x2_0__decpt__7" in mapping_discrete_input_to_discrete_aux["x2"]
    assert "aux_x2_10__decpt__0" in mapping_discrete_input_to_discrete_aux["x2"]

    assert LinearEqualityConstraint(
        features=["aux_x1___neg__0__decpt__3", "aux_x1_5__decpt__0"],
        coefficients=[1] * 2,
        rhs=1,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["aux_x2_0__decpt__7", "aux_x2_10__decpt__0"],
        coefficients=[1] * 2,
        rhs=1,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["x1", "aux_x1___neg__0__decpt__3", "aux_x1_5__decpt__0"],
        coefficients=[1.0] + [0.3, -5.0],
        rhs=0.0,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["x2", "aux_x2_0__decpt__7", "aux_x2_10__decpt__0"],
        coefficients=[1.0] + [-0.7, -10.0],
        rhs=0.0,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])

    assert len(mapped_aux_categorical_inputs) == 3
    assert len(mappings_categorical_var_key_to_aux_var_key_state_pairs["x5"]) == 3

    assert LinearEqualityConstraint(
        features=["aux_x5_a", "aux_x5_b", "aux_x5_c"],
        coefficients=[1] * 3,
        rhs=1,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])

    df_sample = relaxed_domain.inputs.sample(10)
    assert len(df_sample) == 10
    assert len(df_sample.columns) == 11

    df_no_categorical, df_categorical = (
        filter_out_categorical_and_categorical_auxilliary_vars(
            df_sample,
            mapped_aux_categorical_inputs=mapped_aux_categorical_inputs,
            mappings_categorical_var_key_to_aux_var_key_state_pairs=mappings_categorical_var_key_to_aux_var_key_state_pairs,
        )
    )
    assert df_no_categorical.shape == (10, 8)
    assert df_categorical.shape == (10, 1)
    assert (
        pd.testing.assert_frame_equal(
            df_no_categorical[
                [
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "aux_x1___neg__0__decpt__3",
                    "aux_x1_5__decpt__0",
                    "aux_x2_0__decpt__7",
                    "aux_x2_10__decpt__0",
                ]
            ],
            df_sample[
                [
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "aux_x1___neg__0__decpt__3",
                    "aux_x1_5__decpt__0",
                    "aux_x2_0__decpt__7",
                    "aux_x2_10__decpt__0",
                ]
            ],
            check_dtype=False,
        )
        is None
    )
    assert (
        pd.testing.assert_frame_equal(
            df_categorical[["x5"]],
            df_sample[
                [
                    "x5",
                ]
            ],
            check_dtype=False,
        )
        is None
    )

    df_no_discrete_aux = filter_out_discrete_auxilliary_vars(
        df_sample, aux_vars_for_discrete=aux_vars_for_discrete
    )
    assert df_no_discrete_aux.shape == (10, 8)
    assert (
        pd.testing.assert_frame_equal(
            df_no_discrete_aux[
                [
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "x5",
                    "aux_x5_a",
                    "aux_x5_b",
                    "aux_x5_c",
                ]
            ],
            df_sample[
                [
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "x5",
                    "aux_x5_a",
                    "aux_x5_b",
                    "aux_x5_c",
                ]
            ],
            check_dtype=False,
        )
        is None
    )


def test_project_candidates_into_domain_categorical_discrete():
    domain = Domain(
        inputs=Inputs(
            features=[
                DiscreteInput(key="x1", values=[0.3, 5]),
                DiscreteInput(key="x2", values=[0.7, 10]),
                ContinuousInput(key="x3", bounds=[10, 11]),
                ContinuousInput(key="x4", bounds=[5, 11]),
            ]
        ),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(
                    features=["x1", "x2"],
                    coefficients=[-1, -1],
                    rhs=-10,
                ),
                LinearEqualityConstraint(
                    features=["x3", "x4"],
                    coefficients=[1, 1],
                    rhs=15,
                ),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y1")]),
    )
    candidates = pd.DataFrame(
        {
            "x1": [1, 4],
            "x1_aux_1": [0.7, 0.3],
            "x1_aux_2": [0.3, 0.7],
            "x2": [3, 9],
            "x2_aux_1": [0.3, 0.7],
            "x2_aux_2": [0.7, 0.7],
            "x3": [10.1, 10.8],
            "x4": [3, 8.9],
        }
    )
    df = project_candidates_into_domain(
        domain,
        candidates,
        mapping_discrete_input_to_discrete_aux={
            "x1": ["x1_aux_1", "x1_aux_2"],
            "x2": ["x2_aux_1", "x2_aux_2"],
        },
        keys_continuous_inputs=["x3", "x4"],
        scip_params={"parallel/maxnthreads": 1, "numerics/feastol": 1e-8},
    )
    df_true = pd.DataFrame(
        {
            "x1": [0.3, 5.0],
            "x2": [10.0, 10.0],
            "x3": [10.0, 10.0],
            "x4": [5.0, 5.0],
        }
    )
    pd.testing.assert_frame_equal(
        df[["x1", "x2", "x3", "x4"]], df_true[["x1", "x2", "x3", "x4"]]
    )


def test_encode_candidates_to_relaxed_domain():
    """Test encoding candidates from original domain to relaxed domain with auxiliary variables."""
    # Create a domain with categorical, discrete, and continuous inputs
    domain = Domain(
        inputs=Inputs(
            features=[
                CategoricalInput(key="cat1", categories=["A", "B", "C"]),
                DiscreteInput(key="x1", values=[0.3, 5.0, 10.0]),
                ContinuousInput(key="x2", bounds=[0, 10]),
                ContinuousInput(key="x3", bounds=[5, 15]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
    )

    # Create relaxed domain to get the mappings
    (
        relaxed_domain,
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux,
        *_,
    ) = create_continuous_domain(domain=domain)

    # Create candidates in original domain
    candidates = pd.DataFrame(
        {
            "cat1": ["A", "B", "C", "A"],
            "x1": [0.3, 5.0, 10.0, 5.0],
            "x2": [2.5, 7.8, 4.2, 9.1],
            "x3": [6.0, 12.3, 8.5, 14.7],
        }
    )

    # Encode to relaxed domain
    encoded = encode_candidates_to_relaxed_domain(
        candidates=candidates,
        mappings_categorical_var_key_to_aux_var_key_state_pairs=mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux=mapping_discrete_input_to_discrete_aux,
        domain=domain,
    )

    # Check that all auxiliary variables for categoricals are present
    assert "aux_cat1_A" in encoded.columns
    assert "aux_cat1_B" in encoded.columns
    assert "aux_cat1_C" in encoded.columns

    # Check categorical encoding for first row (cat1='A')
    assert encoded.loc[0, "aux_cat1_A"] == 1.0
    assert encoded.loc[0, "aux_cat1_B"] == 0.0
    assert encoded.loc[0, "aux_cat1_C"] == 0.0

    # Check categorical encoding for second row (cat1='B')
    assert encoded.loc[1, "aux_cat1_A"] == 0.0
    assert encoded.loc[1, "aux_cat1_B"] == 1.0
    assert encoded.loc[1, "aux_cat1_C"] == 0.0

    # Check categorical encoding for third row (cat1='C')
    assert encoded.loc[2, "aux_cat1_A"] == 0.0
    assert encoded.loc[2, "aux_cat1_B"] == 0.0
    assert encoded.loc[2, "aux_cat1_C"] == 1.0

    # Check that all auxiliary variables for discrete inputs are present
    assert "aux_x1_0__decpt__3" in encoded.columns
    assert "aux_x1_5__decpt__0" in encoded.columns
    assert "aux_x1_10__decpt__0" in encoded.columns

    # Check discrete encoding for first row (x1=0.3)
    assert encoded.loc[0, "aux_x1_0__decpt__3"] == 1.0
    assert encoded.loc[0, "aux_x1_5__decpt__0"] == 0.0
    assert encoded.loc[0, "aux_x1_10__decpt__0"] == 0.0

    # Check discrete encoding for second row (x1=5.0)
    assert encoded.loc[1, "aux_x1_0__decpt__3"] == 0.0
    assert encoded.loc[1, "aux_x1_5__decpt__0"] == 1.0
    assert encoded.loc[1, "aux_x1_10__decpt__0"] == 0.0

    # Check discrete encoding for third row (x1=10.0)
    assert encoded.loc[2, "aux_x1_0__decpt__3"] == 0.0
    assert encoded.loc[2, "aux_x1_5__decpt__0"] == 0.0
    assert encoded.loc[2, "aux_x1_10__decpt__0"] == 1.0

    # Check that continuous values are preserved
    pd.testing.assert_series_equal(encoded["x2"], candidates["x2"])
    pd.testing.assert_series_equal(encoded["x3"], candidates["x3"])

    # Check that original categorical and discrete columns are still present
    assert "cat1" in encoded.columns
    assert "x1" in encoded.columns


if __name__ == "__main__":
    test_domain_relaxation()
    print("Test passed!")
