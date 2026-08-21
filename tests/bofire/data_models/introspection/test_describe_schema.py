"""Schema-mode navigation: no instance, so paths start at a type name."""

import pytest

from bofire.data_models.introspection.api import describe, full_schema
from bofire.data_models.introspection.paths import PathError
from bofire.data_models.introspection.registry import lookup


def test_class_summary_lists_own_fields():
    result = describe("SoboStrategy")
    assert result["kind"] == "class"
    assert result["name"] == "SoboStrategy"
    # The ``type`` discriminator is implied by ``name`` and is not repeated.
    assert "type" not in result["fields"]
    assert set(result["fields"]) == {
        name for name in lookup("SoboStrategy").model_fields if name != "type"
    }


def test_union_field_is_summarised_by_its_members():
    entry = describe("SoboStrategy")["fields"]["acquisition_function"]
    assert entry["type"] == "union"
    assert "qUCB" in entry["options"]


def test_default_factory_values_are_surfaced():
    """Pydantic omits ``default_factory`` from JSON schema; navigation must not."""
    fields = describe("SoboStrategy")["fields"]
    assert fields["acquisition_function"]["default"] == "qLogNEI"
    assert fields["acquisition_optimizer"]["default"] == "BotorchOptimizer"
    assert fields["surrogate_specs"]["default"] == {"surrogates": []}


def test_union_node():
    result = describe("SoboStrategy.acquisition_function")
    assert result["kind"] == "union"
    assert result["discriminator"] == "type"
    assert "qUCB" in result["options"]


def test_leaf_node():
    result = describe("qUCB.beta")
    assert result["kind"] == "leaf"
    assert result["type"] == "float"


def test_abstract_bases_are_reported_not_offered():
    result = describe("Domain.constraints")
    assert "Constraint" in result["abstract_bases"]
    assert "Constraint" not in result["options"]


def test_keyed_container_without_an_instance_says_keys_need_one():
    result = describe("Domain.inputs")
    assert result["kind"] == "keyed_container"
    assert result["addressing"] == "key"
    assert result["keys"] == "<requires instance>"


def test_crossing_a_union_needs_a_member_named():
    with pytest.raises(PathError, match="is a union"):
        describe("SoboStrategy.acquisition_function.beta")


def test_naming_the_member_crosses_the_union():
    assert describe("SoboStrategy.acquisition_function.qUCB.beta")["kind"] == "leaf"


def test_element_keys_are_unavailable_without_an_instance():
    with pytest.raises(PathError, match="data, not schema"):
        describe("SoboStrategy.domain.inputs.temperature")


def test_descending_past_a_leaf_raises():
    with pytest.raises(PathError, match="leaf"):
        describe("qUCB.beta.nope")


def test_unknown_field_error_lists_the_available_fields():
    with pytest.raises(PathError, match="Fields:"):
        describe("qUCB.nope")


@pytest.mark.parametrize("path", ["StepwiseStrategy.steps", "AdditiveKernel.kernels"])
def test_recursive_models_terminate(path):
    """The walk is lazy, so cycles in the model graph are bounded by path length."""
    assert describe(path)["kind"] == "list"


@pytest.mark.parametrize(
    "name", ["SingleTaskGPSurrogate", "BotorchSurrogates", "Domain", "qUCB"]
)
def test_full_schema_is_shallow_and_navigable(name):
    schema = full_schema(name)
    assert "$defs" not in schema
    assert schema["refs_are_names"] is True


def test_full_schema_refs_resolve_through_the_registry():
    refs = [
        value
        for prop in full_schema("Domain").get("properties", {}).values()
        for key, value in prop.items()
        if key == "$ref"
    ]
    assert refs
    for ref in refs:
        assert lookup(ref)


def test_full_is_rejected_on_a_node_with_no_schema_of_its_own():
    with pytest.raises(ValueError, match="no schema of its own"):
        describe("SoboStrategy.acquisition_function", full=True)


def test_empty_path_without_an_instance_is_an_error():
    with pytest.raises(PathError, match="type name"):
        describe("")
