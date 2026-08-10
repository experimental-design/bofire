import pytest

from bofire.data_models.features.api import CategoricalTaskInput, ContinuousTaskInput


def test_validate_fidelities_default_generation():
    feat = CategoricalTaskInput(
        key="task",
        categories=["p1", "p2"],
    )
    assert feat.fidelities == [0, 0]


# rejection of descriptor data on task inputs is covered by the add_invalid specs
# (`descriptors` is narrowed to None, so the type rejects it); this covers the accepted side.
TASK_CASES = [
    (CategoricalTaskInput, {"categories": ["p1", "p2"]}),
    (ContinuousTaskInput, {"bounds": (0, 1)}),
]
IDS = ["categorical", "continuous"]


@pytest.mark.parametrize("cls, kwargs", TASK_CASES, ids=IDS)
def test_task_input_carries_no_descriptors(cls, kwargs):
    feat = cls(key="task", **kwargs)
    assert feat.descriptors is None
    # and the constraint is in the type, so it shows up in the schema
    assert cls.model_fields["descriptors"].annotation is type(None)
