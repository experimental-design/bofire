"""Instance-mode navigation: paths are relative to a spec, keys resolve, values show."""

import pytest

from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.introspection.api import describe
from bofire.data_models.introspection.paths import PathError
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import SoboStrategy


@pytest.fixture
def strategy() -> SoboStrategy:
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 80), unit="degC"),
                ContinuousInput(key="pressure", bounds=(1, 10)),
                CategoricalInput(key="solvent", categories=["water", "ethanol"]),
                # A key containing the path separator is legal and must round-trip.
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
    return SoboStrategy(domain=domain)


def test_keyed_container_lists_its_keys(strategy):
    result = describe("domain.inputs", instance=strategy)
    assert result["kind"] == "keyed_container"
    assert result["keys"] == {
        "temperature": "ContinuousInput",
        "pressure": "ContinuousInput",
        "solvent": "CategoricalInput",
        "a.b": "ContinuousInput",
    }


def test_leaf_reports_its_value_and_constraints(strategy):
    result = describe("domain.inputs.temperature.bounds", instance=strategy)
    assert result["kind"] == "leaf"
    assert result["value"] == [20.0, 80.0]
    assert result["constraints"] == {"min_length": 2, "max_length": 2}


def test_key_containing_a_dot_is_reachable_when_quoted(strategy):
    result = describe('domain.inputs["a.b"].bounds', instance=strategy)
    assert result["value"] == [0.0, 1.0]


def test_an_instance_collapses_a_union_to_the_member_present(strategy):
    result = describe("acquisition_function", instance=strategy)
    assert result["kind"] == "class"
    assert result["is"] == "qLogNEI"
    # The menu is still reported: knowing what one could switch to is the point.
    assert "qUCB" in result["options"]
    assert "n_mc_samples" in result["fields"]


def test_unkeyed_container_is_index_addressed(strategy):
    result = describe("domain.constraints", instance=strategy)
    assert result["kind"] == "list"
    assert result["addressing"] == "index"
    assert result["length"] == 1
    assert result["items"][0]["is"] == "LinearInequalityConstraint"
    # ``to_description`` is reused rather than inventing a second prose channel.
    assert "temperature" in result["items"][0]["summary"]


def test_index_addressed_element(strategy):
    result = describe("domain.constraints[0].rhs", instance=strategy)
    assert result["value"] == 70.0


@pytest.mark.parametrize(
    "path",
    [
        "",
        "domain",
        "domain.inputs",
        "domain.inputs.temperature",
        "domain.inputs.temperature.bounds",
        "domain.outputs.yield.objective",
        "domain.constraints",
        "domain.constraints[0]",
        "acquisition_function",
        "surrogate_specs",
        "folds",
    ],
)
def test_object_and_dict_instances_describe_identically(strategy, path):
    """An agent holding a serialized spec must get the same answer as one holding an object."""
    assert describe(path, instance=strategy) == describe(
        path, instance=strategy.model_dump()
    )


def test_a_dict_without_a_discriminator_needs_an_explicit_root(strategy):
    spec = strategy.model_dump()
    spec.pop("type")
    with pytest.raises(PathError, match="root="):
        describe("folds", instance=spec)
    assert describe("folds", instance=spec, root="SoboStrategy")["value"] == 5


def test_unknown_key_error_lists_the_available_keys(strategy):
    with pytest.raises(PathError, match="Keys: temperature"):
        describe("domain.inputs.nope", instance=strategy)


def test_unkeyed_container_error_points_at_index_addressing(strategy):
    with pytest.raises(PathError, match=r"index-addressed"):
        describe("domain.constraints.foo", instance=strategy)


def test_paths_reported_by_describe_are_reusable(strategy):
    """A path emitted here must parse back — the write path will consume it verbatim."""
    emitted = describe('domain.inputs["a.b"]', instance=strategy)["path"]
    assert describe(emitted, instance=strategy)["name"] == "ContinuousInput"


class TestFieldNameShadowsElementKey:
    """A feature may legally be keyed after one of its container's own fields."""

    @pytest.fixture
    def domain(self) -> Domain:
        return Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="features", bounds=(0, 1)),
                    ContinuousInput(key="x", bounds=(0, 1)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MaximizeObjective())]
            ),
        )

    def test_the_collision_is_reported(self, domain):
        assert describe("inputs", instance=domain)["shadowed_keys"] == ["features"]

    def test_the_declared_field_wins(self, domain):
        assert describe("inputs.features", instance=domain)["kind"] == "list"

    def test_going_through_the_list_field_reaches_the_element(self, domain):
        result = describe("inputs.features.features", instance=domain)
        assert result["name"] == "ContinuousInput"
        assert result["fields"]["key"]["value"] == "features"


def test_every_valid_strategy_spec_is_navigable(strategy_spec):
    """Sweep over the existing spec fixtures: the root and each field must resolve."""
    obj = strategy_spec.obj()
    root = describe("", instance=obj)
    assert root["kind"] in ("class", "keyed_container", "list")
    for name in root.get("fields", {}):
        describe(name, instance=obj)


def test_every_valid_domain_spec_is_navigable(domain_spec):
    obj = domain_spec.obj()
    for name in describe("", instance=obj).get("fields", {}):
        describe(name, instance=obj)
