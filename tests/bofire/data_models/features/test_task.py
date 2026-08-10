import pytest

from bofire.data_models.features.api import CategoricalTaskInput, ContinuousTaskInput


def test_validate_fidelities_default_generation():
    feat = CategoricalTaskInput(
        key="task",
        categories=["p1", "p2"],
    )
    assert feat.fidelities == [0, 0]


# a categorical task has one descriptor level per category, a continuous one a single
# level; the payloads below are sized accordingly so the *task* rule is what rejects
# them, rather than the per-level length check firing first.
TASK_CASES = [
    (CategoricalTaskInput, {"categories": ["p1", "p2"]}, [1.0, 10.0], ["O", "CCO"]),
    (ContinuousTaskInput, {"bounds": (0, 1)}, [1.0], ["CCO"]),
]
IDS = ["categorical", "continuous"]


@pytest.mark.parametrize("cls, kwargs, _values, _structure", TASK_CASES, ids=IDS)
def test_task_input_accepts_no_descriptor_data(cls, kwargs, _values, _structure):
    """The empty sentinels of both inherited fields are fine."""
    feat = cls(key="task", **kwargs)
    assert feat.descriptors == {}
    assert feat.structure is None


# construction-time rejection is covered by the add_invalid specs; assignment is not,
# and BaseModel sets validate_assignment=True, so the guard must hold there too.
@pytest.mark.parametrize("cls, kwargs, values, _structure", TASK_CASES, ids=IDS)
def test_task_input_rejects_descriptors_on_assignment(cls, kwargs, values, _structure):
    feat = cls(key="task", **kwargs)
    with pytest.raises(ValueError, match="task inputs cannot carry `descriptors`"):
        feat.descriptors = {"cost": values}


@pytest.mark.parametrize("cls, kwargs, _values, structure", TASK_CASES, ids=IDS)
def test_task_input_rejects_structure_on_assignment(cls, kwargs, _values, structure):
    feat = cls(key="task", **kwargs)
    with pytest.raises(ValueError, match="task inputs cannot carry a `structure`"):
        feat.structure = structure


@pytest.mark.parametrize("cls, kwargs, _values, _structure", TASK_CASES, ids=IDS)
def test_task_input_empty_structure_reports_length_not_task_rule(
    cls, kwargs, _values, _structure
):
    """An empty list carries no structure, so the guard (a falsy check) lets it pass
    to the per-level length check, which is the accurate complaint."""
    with pytest.raises(ValueError, match=r"structure must have \d+ value"):
        cls(key="task", structure=[], **kwargs)
