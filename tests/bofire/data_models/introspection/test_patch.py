"""Patching a serialized spec: the write half of the path vocabulary."""

import copy
import json

import pytest
from pydantic import ValidationError

from bofire.data_models.domain.api import Domain
from bofire.data_models.introspection.api import describe, reconfigure
from bofire.data_models.introspection.patch import PatchError, reconfigure_json
from bofire.data_models.introspection.paths import PathError
from bofire.data_models.objectives.api import MaximizeObjective


# The reconfigure verbs from the Sara/lenz paper, expressed as paths. The last
# entry is model selection, which lenz has no command for at all.
LENZ_VERBS = [
    ("set-acqf", "acquisition_function", {"type": "qUCB", "beta": 2.0}),
    ("set-bounds", "domain.inputs.temperature.bounds", [60.0, 90.0]),
    ("set-constraints", "domain.constraints", []),
    (
        "set-surrogate",
        "surrogate_specs.yield.kernel",
        {"type": "MaternKernel", "nu": 1.5, "ard": True},
    ),
]


@pytest.mark.parametrize(
    "verb, path, value", LENZ_VERBS, ids=[v[0] for v in LENZ_VERBS]
)
def test_lenz_reconfigure_verbs(spec, verb, path, value):
    result = reconfigure(spec, [{"path": path, "value": value}], verb)
    assert result["applied"] == [path]
    # The result is a valid spec that describe() can navigate straight away.
    assert describe(path, instance=result["spec"])


def test_promoting_an_outcome_to_an_objective(domain):
    """The paper's mid-run reformulation, in one call.

    ``cost`` starts as a constraint-shaped objective and is promoted to a real
    one. That changes which outputs ``MoboStrategy.ref_point`` is derived from,
    so the promotion alone leaves the spec invalid -- the resync pass has to
    regenerate the reference point along with it.
    """
    from bofire.data_models.domain.api import Outputs
    from bofire.data_models.features.api import ContinuousOutput
    from bofire.data_models.objectives.api import (
        MinimizeObjective,
        MinimizeSigmoidObjective,
    )
    from bofire.data_models.strategies.api import MoboStrategy

    constrained = Outputs(
        features=[
            ContinuousOutput(key="yield", objective=MaximizeObjective()),
            ContinuousOutput(key="throughput", objective=MaximizeObjective()),
            ContinuousOutput(
                key="cost",
                objective=MinimizeSigmoidObjective(tp=100.0, steepness=1.0),
            ),
        ]
    )
    spec = MoboStrategy(
        domain=Domain(inputs=domain.inputs, outputs=constrained)
    ).model_dump(mode="json")
    assert set(spec["ref_point"]["values"]) == {"yield", "throughput"}

    result = reconfigure(
        spec,
        [
            {
                "path": "domain.outputs.cost.objective",
                "value": MinimizeObjective().model_dump(mode="json"),
            }
        ],
        "requirements changed: map the full cost trade-off",
    )
    assert result["reset"] == ["ref_point"]
    assert set(result["spec"]["ref_point"]["values"]) == {"yield", "throughput", "cost"}


def test_changes_apply_in_order_against_the_evolving_copy(spec):
    result = reconfigure(
        spec,
        [
            {"path": "folds", "value": 7},
            {"path": "domain.inputs.temperature.bounds", "value": [60.0, 90.0]},
            {"path": "acquisition_function", "value": {"type": "qUCB", "beta": 2.0}},
        ],
        "batch",
    )
    assert describe("folds", instance=result["spec"])["value"] == 7
    assert result["summary"]["acquisition_function"] == "qUCB"


def test_input_spec_is_never_mutated(spec):
    before = copy.deepcopy(spec)
    reconfigure(spec, [{"path": "folds", "value": 99}], "why")
    assert spec == before


def test_a_failing_batch_leaves_the_input_untouched(spec):
    before = copy.deepcopy(spec)
    with pytest.raises(ValidationError):
        reconfigure(
            spec,
            [
                {"path": "folds", "value": 7},
                {"path": "acquisition_function", "value": {"type": "qFoo"}},
            ],
            "second change is bad",
        )
    assert spec == before


def test_unknown_tag_error_names_the_expected_tags(spec):
    with pytest.raises(
        ValidationError, match="does not match any of the expected tags"
    ):
        reconfigure(
            spec, [{"path": "acquisition_function", "value": {"type": "qFoo"}}], "bad"
        )


def test_error_loc_reaches_the_offending_leaf(spec):
    with pytest.raises(ValidationError) as excinfo:
        reconfigure(
            spec,
            [
                {
                    "path": "acquisition_function",
                    "value": {"type": "qUCB", "beta": "hot"},
                }
            ],
            "bad",
        )
    # The rebuild validates the whole tree with discriminators intact, so the
    # location is precise rather than one error per union member.
    assert excinfo.value.errors()[0]["loc"] == ("acquisition_function", "qUCB", "beta")


def test_domain_level_validators_still_fire(spec):
    with pytest.raises(ValidationError, match="monotonically increasing"):
        reconfigure(
            spec,
            [{"path": "domain.inputs.temperature.bounds", "value": [90.0, 60.0]}],
            "inverted",
        )


@pytest.mark.parametrize(
    "changes, error",
    [
        ([{"value": 1}], PatchError),
        ([{"path": "folds"}], PatchError),
        ([{"path": 1, "value": 1}], PatchError),
        ([{"path": "folds", "value": 1, "extra": 2}], PatchError),
        (["folds"], PatchError),
        ([{"path": "nope", "value": 1}], PathError),
        ([{"path": "domain.inputs.nope.bounds", "value": [0, 1]}], PathError),
    ],
)
def test_malformed_changes_are_rejected(spec, changes, error):
    with pytest.raises(error):
        reconfigure(spec, changes, "bad")


def test_spec_must_carry_a_type_discriminator():
    with pytest.raises(PatchError, match="discriminator"):
        reconfigure({"folds": 5}, [], "no type")


def test_result_is_json_serialisable_and_a_fixpoint(spec):
    first = reconfigure(spec, [{"path": "folds", "value": 7}], "once")["spec"]
    json.dumps(first)
    again = reconfigure(first, [], "no-op")["spec"]
    assert again == first


def test_no_op_reconfigure_returns_the_canonicalised_spec(strategy_spec):
    """Sweep the existing spec fixtures: every valid strategy survives a no-op."""
    obj = strategy_spec.obj()
    dumped = obj.model_dump(mode="json")
    assert reconfigure(dumped, [], "no-op")["spec"] == dumped


def test_summary_is_small_and_useful(spec):
    summary = reconfigure(spec, [], "no-op")["summary"]
    assert summary["type"] == "SoboStrategy"
    assert summary["objectives"] == {"yield": "MaximizeObjective"}
    assert summary["surrogates"] == {"yield": "SingleTaskGPSurrogate"}
    # Small enough to return on every call; the full spec is ~700+ tokens.
    assert len(json.dumps(summary)) < 400


def test_root_replacement(spec):
    other = copy.deepcopy(spec)
    other["folds"] = 3
    assert (
        reconfigure(spec, [{"path": "", "value": other}], "swap")["spec"]["folds"] == 3
    )


def test_json_string_wrapper(spec):
    out = json.loads(
        reconfigure_json(
            json.dumps(spec), json.dumps([{"path": "folds", "value": 8}]), "s"
        )
    )
    assert out["spec"]["folds"] == 8
    assert out["rationale"] == "s"
