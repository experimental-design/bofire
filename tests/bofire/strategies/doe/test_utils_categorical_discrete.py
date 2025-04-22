import pandas as pd

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.doe.utils_categorical_discrete import (
    project_candidates_into_domain,
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
    print("Test passed!")
