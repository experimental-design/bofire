from bofire.data_models.features.api import TaskInput


def test_validate_fidelities_default_generation():
    feat = TaskInput(
        key="task",
        categories=["p1", "p2"],
    )
    assert feat.fidelities == [0, 0]
