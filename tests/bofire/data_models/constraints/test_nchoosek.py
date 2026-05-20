from bofire.data_models.constraints.api import NChooseKConstraint


def test_nchoosek_to_description():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    assert c.to_description() == "Choose 1-2 active features from ['x1', 'x2', 'x3']"


def test_nchoosek_to_description_none_valid():
    c = NChooseKConstraint(
        features=["x1", "x2"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    assert c.to_description() == "Choose 1-2 active features from ['x1', 'x2'], or none"


def test_nchoosek_count_is_valid():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    assert c.count_is_valid(1) is True
    assert c.count_is_valid(2) is True
    assert c.count_is_valid(0) is False
    assert c.count_is_valid(3) is False


def test_nchoosek_count_is_valid_none_also_valid():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    # none_also_valid carves out count==0 even though min_count > 0
    assert c.count_is_valid(0) is True
    # max_count is still a hard ceiling
    assert c.count_is_valid(3) is False
