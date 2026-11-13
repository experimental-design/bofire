from pydantic import TypeAdapter

from bofire.data_models.objectives.api import (
    AnyObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
)


def test_stored_neutral_weight_is_dropped():
    # objectives stored before the weights moved to the scalarizations carry w=1.0
    stored = {"type": "MaximizeSigmoidObjective", "w": 1.0, "tp": 0.5, "steepness": 2}
    objective = TypeAdapter(AnyObjective).validate_python(stored)
    assert objective == MaximizeSigmoidObjective(tp=0.5, steepness=2)
    assert "w" not in objective.model_dump()
    assert MaximizeObjective(w=1) == MaximizeObjective()  # ty: ignore[unknown-argument]
