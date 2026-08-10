import pytest

from bofire.data_models.features.api import CategoricalTaskInput, ContinuousTaskInput


def test_validate_fidelities_default_generation():
    feat = CategoricalTaskInput(
        key="task",
        categories=["p1", "p2"],
    )
    assert feat.fidelities == [0, 0]


# rejection of descriptor data on task inputs is covered by the add_invalid specs;
# these cover what a spec cannot express as neatly.
TASK_CASES = [
    (CategoricalTaskInput, {"categories": ["p1", "p2"]}),
    (ContinuousTaskInput, {"bounds": (0, 1)}),
]
IDS = ["categorical", "continuous"]


@pytest.mark.parametrize("cls, kwargs", TASK_CASES, ids=IDS)
def test_task_input_accepts_no_descriptor_data(cls, kwargs):
    """The empty sentinels of both inherited fields are fine."""
    feat = cls(key="task", **kwargs)
    assert feat.descriptors == {}
    assert feat.structure is None


@pytest.mark.parametrize("cls, kwargs", TASK_CASES, ids=IDS)
def test_task_input_empty_structure_reports_length_not_task_rule(cls, kwargs):
    """An empty list carries no structure, so the guard (a falsy check) lets it pass
    to the per-level length check, which is the accurate complaint."""
    with pytest.raises(ValueError, match=r"structure must have \d+ value"):
        cls(key="task", structure=[], **kwargs)
