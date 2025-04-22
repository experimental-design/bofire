import pandas as pd

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.doe.utils_categorical_discrete import (
    create_continuous_domain,
    project_candidates_into_domain,
)


def test_domain_relaxation():
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
    (
        relaxed_domain,
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux,
        aux_vars_for_discrete,
        mapped_aux_categorical_inputs,
        mapped_continous_inputs,
    ) = create_continuous_domain(domain=domain)

    assert len(relaxed_domain.inputs.get([DiscreteInput])) == 0
    assert len(relaxed_domain.inputs.get([ContinuousInput])) == 8
    assert len(relaxed_domain.constraints.get([LinearInequalityConstraint])) == 1
    assert len(relaxed_domain.constraints.get([LinearEqualityConstraint])) == 5
    assert len(relaxed_domain.constraints.get([NChooseKConstraint])) == 0

    assert len(mappings_categorical_var_key_to_aux_var_key_state_pairs) == 0
    assert len(mapping_discrete_input_to_discrete_aux) == 2
    assert len(aux_vars_for_discrete) == 4

    for x in ["x1", "x2"]:
        assert len(mapping_discrete_input_to_discrete_aux[x]) == 2

    assert len(mapped_continous_inputs) == 2

    assert "aux_x1_0_3" in mapping_discrete_input_to_discrete_aux["x1"]
    assert "aux_x1_5_0" in mapping_discrete_input_to_discrete_aux["x1"]

    assert "aux_x2_0_7" in mapping_discrete_input_to_discrete_aux["x2"]
    assert "aux_x2_10_0" in mapping_discrete_input_to_discrete_aux["x2"]

    assert LinearEqualityConstraint(
        features=["aux_x1_0_3", "aux_x1_5_0"],
        coefficients=[1] * 2,
        rhs=1,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["aux_x2_0_7", "aux_x2_10_0"],
        coefficients=[1] * 2,
        rhs=1,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["x1", "aux_x1_0_3", "aux_x1_5_0"],
        coefficients=[1.0] + [-0.3, -5.0],
        rhs=0.0,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])
    assert LinearEqualityConstraint(
        features=["x2", "aux_x2_0_7", "aux_x2_10_0"],
        coefficients=[1.0] + [-0.7, -10.0],
        rhs=0.0,
    ) in relaxed_domain.constraints.get([LinearEqualityConstraint])


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
        scip_params={"parallel/maxnthreads": 1},
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


if __name__ == "__main__":
    test_project_candidates_into_domain_categorical_discrete()
    test_domain_relaxation()
    print("Test passed!")
