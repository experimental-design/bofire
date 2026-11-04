"""Navigation alongside a serialized spec: keys resolve and values are reported."""

import pytest

from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.introspection.api import PathError, describe
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import SoboStrategy


@pytest.fixture
def spec() -> dict:
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 80)),
                ContinuousInput(key="pressure", bounds=(1, 10)),
                # A key containing a dot needs no escaping in a list path.
                ContinuousInput(key="a.b", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="yield", objective=MaximizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(
                    features=["temperature", "pressure"], coefficients=[1, 1], rhs=70
                )
            ]
        ),
    )
    return SoboStrategy(domain=domain).model_dump(mode="json")


def test_root(spec):
    result = describe([], spec=spec)
    assert result["kind"] == "class"
    assert result["fields"]["include_infeasible_exps_in_acqf_calc"]["value"] is False
    assert result["fields"]["acquisition_function"]["is"] == "qLogNEI"


def test_keyed_container_lists_its_keys(spec):
    result = describe(["domain", "inputs"], spec=spec)
    assert result["addressing"] == "key"
    assert result["keys"] == {
        "temperature": "ContinuousInput",
        "pressure": "ContinuousInput",
        "a.b": "ContinuousInput",
    }


@pytest.mark.parametrize(
    "path, value",
    [
        (["domain", "inputs", "temperature", "bounds"], [20.0, 80.0]),
        (["domain", "inputs", "a.b", "bounds"], [0.0, 1.0]),
        # The long form through the list field resolves like the shortcut.
        (["domain", "inputs", "features", "temperature", "bounds"], [20.0, 80.0]),
        (["domain", "inputs", 0, "bounds"], [20.0, 80.0]),
        (["domain", "constraints", 0, "rhs"], 70.0),
        (["acquisition_function", "n_mc_samples"], 512),
        # Surrogates are addressed by the output they model.
        (["surrogate_specs", "yield", "inputs", "pressure", "bounds"], [1.0, 10.0]),
        (
            ["surrogate_specs", "surrogates", "yield", "inputs", "pressure", "bounds"],
            [1.0, 10.0],
        ),
    ],
)
def test_leaf_values(spec, path, value):
    result = describe(path, spec=spec)
    assert result["kind"] == "leaf"
    assert result["value"] == value


def test_list_of_plain_values_is_a_leaf(spec):
    result = describe(["domain", "constraints", 0, "features"], spec=spec)
    assert result["value"] == ["temperature", "pressure"]


def test_union_reports_the_member_present(spec):
    result = describe(["domain", "outputs", "yield", "objective"], spec=spec)
    assert result["kind"] == "union"
    assert result["is"] == "MaximizeObjective"


def test_unkeyed_container_reports_its_length(spec):
    result = describe(["domain", "constraints"], spec=spec)
    assert result["addressing"] == "index"
    assert result["length"] == 1


@pytest.mark.parametrize(
    "path, match",
    [
        (["domain", "inputs", "nope"], "Keys: \\['temperature'"),
        (["domain", "constraints", "foo"], "addressed by index"),
        (["domain", "constraints", 5], "out of range"),
        (["include_infeasible_exps_in_acqf_calc", "x"], "leaf"),
    ],
)
def test_bad_paths(spec, path, match):
    with pytest.raises(PathError, match=match):
        describe(path, spec=spec)


def test_a_spec_without_type_needs_a_root(spec):
    spec.pop("type")
    with pytest.raises(PathError, match="root="):
        describe(["seed"], spec=spec)
    assert describe(["seed"], spec=spec, root="SoboStrategy")["kind"] == "leaf"


def test_spec_must_be_a_dict(spec):
    with pytest.raises(PathError, match="dict"):
        describe([], spec=SoboStrategy(**spec))


def test_a_declared_field_shadows_an_element_key_of_the_same_name():
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="features", bounds=(0, 1)),
                ContinuousInput(key="x", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
    ).model_dump(mode="json")
    assert describe(["inputs", "features"], spec=domain)["kind"] == "list"
    # Going through the list field reaches the element.
    assert describe(["inputs", "features", "features"], spec=domain)["is"] == (
        "ContinuousInput"
    )
    assert (
        describe(["inputs", "features", "features", "key"], spec=domain)["value"]
        == "features"
    )


def test_every_valid_strategy_spec_is_navigable(strategy_spec):
    """Sweep the existing spec fixtures: the root and each field resolve."""
    spec = strategy_spec.obj().model_dump(mode="json")
    root = describe([], spec=spec)
    for name in root["fields"]:
        describe([name], spec=spec)
