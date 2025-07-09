import pytest

from bofire.data_models.constraints.api import (
    CategoricalExcludeConstraint,
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)


def test_validate_inputs():
    inputs = Inputs(
        features=[
            ContinuousInput(key="temperature", bounds=[0, 100]),
            CategoricalInput(key="solvent", categories=["Acetone", "THF"]),
        ],
    )
    c = CategoricalExcludeConstraint(
        features=["solvent", "catalyst"],
        conditions=[
            SelectionCondition(
                selection=["Acetone", "THF"],
            ),
            SelectionCondition(
                selection=["alpha", "beta"],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Feature catalyst is not a input feature in the provided Inputs object.",
    ):
        c.validate_inputs(inputs)

    inputs = Inputs(
        features=[
            CategoricalInput(key="catalyst", categories=["alpha", "beta"]),
            CategoricalInput(key="solvent", categories=["Acetone", "THF"]),
        ],
    )
    c = CategoricalExcludeConstraint(
        features=["solvent", "catalyst"],
        conditions=[
            ThresholdCondition(
                threshold=50,
                operator=">",
            ),
            SelectionCondition(
                selection=["alpha", "beta"],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Condition for feature solvent is not a SubSelectionCondition.",
    ):
        c.validate_inputs(inputs)

    inputs = Inputs(
        features=[
            CategoricalInput(key="catalyst", categories=["alpha", "beta"]),
            CategoricalInput(key="solvent", categories=["Acetone", "THF"]),
        ],
    )

    c = CategoricalExcludeConstraint(
        features=["solvent", "catalyst"],
        conditions=[
            SelectionCondition(
                selection=["Acetone", "THF", "water"],
            ),
            SelectionCondition(
                selection=["alpha", "beta"],
            ),
        ],
    )
    with pytest.raises(
        ValueError,
        match="Some categories in condition 0 are not valid categories for feature solvent.",
    ):
        c.validate_inputs(inputs)

    inputs = Inputs(
        features=[
            CategoricalInput(key="catalyst", categories=["alpha", "beta"]),
            DiscreteInput(key="temperature", values=[10, 20]),
        ],
    )
    c = CategoricalExcludeConstraint(
        features=["temperature", "catalyst"],
        conditions=[
            SelectionCondition(
                selection=[30],
            ),
            SelectionCondition(
                selection=["alpha", "beta"],
            ),
        ],
    )
    with pytest.raises(
        ValueError,
        match="Some values in condition 0 are not valid values for feature temperature.",
    ):
        c.validate_inputs(inputs)

    inputs = Inputs(
        features=[
            DiscreteInput(key="pressure", values=[5, 10]),
            DiscreteInput(key="temperature", values=[10, 20]),
        ],
    )
    c = CategoricalExcludeConstraint(
        features=["temperature", "pressure"],
        conditions=[
            SelectionCondition(
                selection=[10],
            ),
            SelectionCondition(
                selection=[10],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="At least one of the features must be a CategoricalInput feature.",
    ):
        c.validate_inputs(inputs)

    inputs = Inputs(
        features=[
            CategoricalInput(key="catalyst", categories=["alpha", "beta"]),
            ContinuousInput(key="temperature", bounds=[0, 100]),
        ],
    )
    c = CategoricalExcludeConstraint(
        features=["temperature", "catalyst"],
        conditions=[
            SelectionCondition(
                selection=[30],
            ),
            SelectionCondition(
                selection=["alpha", "beta"],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Condition for ContinuousInput temperature is not a ThresholdCondition.",
    ):
        c.validate_inputs(inputs)
