from bofire.data_models.constraints.api import NChooseKConstraint


def test_nchoosek_to_description():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    desc = c.to_description()
    assert "Choose 1-2" in desc
    assert "x1" in desc


def test_nchoosek_to_description_none_valid():
    c = NChooseKConstraint(
        features=["x1", "x2"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    assert "or none" in c.to_description()
