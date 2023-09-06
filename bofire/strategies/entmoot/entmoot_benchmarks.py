from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective


def build_multi_obj_categorical_problem(n_obj: int = 2, no_cat=False) -> Domain:
    """
    Builds a small test example which is used in Entmoot tests.
    """

    cat_feat = (
        []
        if no_cat
        else [CategoricalInput(key="x0", categories=("blue", "orange", "gray"))]
    )
    input_features = Inputs(
        features=cat_feat
        + [
            DiscreteInput(key="x1", values=[5, 6]),
            DiscreteInput(key="x2", values=[0, 1]),  # binary
            ContinuousOutput(key="x3", values=[5.0, 6.0]),
            ContinuousOutput(key="x4", values=[4.6, 6.0]),
            ContinuousOutput(key="x5", values=[5.0, 6.0]),
        ]
    )

    output_features = Outputs(
        features=[
            ContinuousOutput(
                key=f"y{i}", objective=MaximizeObjective(w=1.0, bounds=[0.0, 1.0])
            )
            for i in range(n_obj)
        ]
    )

    domain = Domain(inputs=input_features, outputs=output_features)

    return domain
