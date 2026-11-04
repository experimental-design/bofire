"""Schema-only navigation: no spec, so a path starts at a type name."""

import pytest

from bofire.data_models.introspection.api import (
    PathError,
    UnknownNameError,
    describe,
    full_schema,
    lookup,
)


def test_class_summary_lists_own_fields_without_the_discriminator():
    result = describe(["SoboStrategy"])
    assert result["kind"] == "class"
    assert result["path"] == ["SoboStrategy"]
    assert set(result["fields"]) == {
        name for name in lookup("SoboStrategy").model_fields if name != "type"
    }


def test_union_field_lists_its_members():
    entry = describe(["SoboStrategy"])["fields"]["acquisition_function"]
    assert entry["type"] == "union"
    assert "qUCB" in entry["options"]


def test_default_factory_values_are_surfaced():
    """Pydantic omits ``default_factory`` values from JSON schema."""
    fields = describe(["SoboStrategy"])["fields"]
    assert fields["acquisition_function"]["default"] == "qLogNEI"
    assert fields["acquisition_optimizer"]["default"] == "BotorchOptimizer"


def test_union_node():
    result = describe(["SoboStrategy", "acquisition_function"])
    assert result["kind"] == "union"
    assert result["discriminator"] == "type"
    assert "qUCB" in result["options"]


def test_naming_a_member_crosses_the_union():
    result = describe(["SoboStrategy", "acquisition_function", "qUCB", "beta"])
    assert result["kind"] == "leaf"
    assert result["type"] == "float"


def test_a_member_is_found_by_tag_as_well_as_class_name():
    # GeneticAlgorithmOptimizer's discriminator tag is "GeneticAlgorithm".
    for name in ("GeneticAlgorithm", "GeneticAlgorithmOptimizer"):
        path = ["SoboStrategy", "acquisition_optimizer", name]
        assert describe(path)["name"] == "GeneticAlgorithmOptimizer"


def test_constraints_are_index_addressed_and_abstract_base_is_not_offered():
    result = describe(["Domain", "constraints"])
    assert result["addressing"] == "index"
    assert "LinearInequalityConstraint" in result["options"]
    assert "Constraint" not in result["options"]


def test_keyed_container():
    result = describe(["Domain", "inputs"])
    assert result["kind"] == "list"
    assert result["addressing"] == "key"
    assert "keys" not in result  # keys are data; there is no spec


def test_enum_leaf_lists_its_values():
    result = describe(["SingleTaskGPSurrogate", "output_scaler"])
    assert "STANDARDIZE" in result["values"]


@pytest.mark.parametrize(
    "path, match",
    [
        (["SoboStrategy", "acquisition_function", "beta"], "is a union"),
        (["SoboStrategy", "domain", "inputs", "temperature"], "not schema"),
        (["qUCB", "beta", "nope"], "leaf"),
        (["qUCB", "nope"], "Fields:"),
        ([], "type name"),
        ([0], "type name"),
        ("SoboStrategy", "list of segments"),
        (["qUCB", 1.5], "strings or integers"),
        (["qUCB", True], "strings or integers"),
    ],
)
def test_bad_paths_raise_with_a_useful_message(path, match):
    with pytest.raises(PathError, match=match):
        describe(path)


def test_unknown_type_name():
    with pytest.raises(UnknownNameError):
        describe(["NoSuchModel"])


@pytest.mark.parametrize(
    "path", [["StepwiseStrategy", "steps"], ["AdditiveKernel", "kernels"]]
)
def test_recursive_models_terminate(path):
    assert describe(path)["kind"] == "list"


@pytest.mark.parametrize("name", ["SingleTaskGPSurrogate", "Domain", "qUCB"])
def test_full_schema_is_shallow(name):
    schema = describe([name], full=True)
    assert "$defs" not in schema
    assert schema["refs_are_names"] is True
    assert schema == {"path": [name], **full_schema(name)}


def test_full_schema_refs_are_describable_names():
    refs = [
        prop["$ref"]
        for prop in full_schema("Domain")["properties"].values()
        if "$ref" in prop
    ]
    assert refs
    for ref in refs:
        assert describe([ref])["kind"] in ("class", "list")


def test_full_is_rejected_on_a_union():
    with pytest.raises(ValueError, match="no schema of its own"):
        describe(["SoboStrategy", "acquisition_function"], full=True)
