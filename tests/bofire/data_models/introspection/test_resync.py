"""Keeping derived copies consistent when the domain is patched.

A spec duplicates domain data, so a patch to `domain` alone leaves stale copies.
`_check_compability` catches some of that loudly; `StepwiseStrategy` does not
catch it at all, which is the case these tests exist for.
"""

import pytest

from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.introspection.api import describe, reconfigure
from bofire.data_models.introspection.resync import ResyncError
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    MoboStrategy,
    NumberOfExperimentsCondition,
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)


NARROW = [{"path": "domain.inputs.temperature.bounds", "value": [60.0, 90.0]}]


def bounds_at(spec, path):
    return describe(path, instance=spec)["value"]


def test_a_domain_patch_alone_would_be_rejected(spec):
    """The control: without propagation, the rebuild fails.

    This is what makes the resync pass necessary rather than cosmetic.
    """
    import copy

    from pydantic import ValidationError

    stale = copy.deepcopy(spec)
    stale["domain"]["inputs"]["features"][0]["bounds"] = [60.0, 90.0]
    with pytest.raises(ValidationError, match="incompatible"):
        SoboStrategy(**stale)


def test_parametric_patch_propagates_into_the_surrogate(spec):
    result = reconfigure(spec, NARROW, "narrow to the productive region")
    assert result["propagated"] == ["surrogate_specs.surrogates[0].inputs.features[0]"]
    assert bounds_at(result["spec"], "domain.inputs.temperature.bounds") == [60.0, 90.0]
    assert bounds_at(
        result["spec"], "surrogate_specs.yield.inputs.temperature.bounds"
    ) == [60.0, 90.0]


def test_propagation_preserves_customised_surrogate_config(custom_spec):
    """Propagation replaces the stale feature, not the surrogate around it."""
    result = reconfigure(custom_spec, NARROW, "narrow")
    assert (
        describe("surrogate_specs.yield.kernel.ard", instance=result["spec"])["value"]
        is False
    )
    assert bounds_at(
        result["spec"], "surrogate_specs.yield.inputs.temperature.bounds"
    ) == [60.0, 90.0]


def test_untouched_features_are_not_propagated(spec):
    result = reconfigure(spec, NARROW, "narrow")
    assert bounds_at(
        result["spec"], "surrogate_specs.yield.inputs.pressure.bounds"
    ) == [
        1.0,
        10.0,
    ]


def test_a_feature_duplicated_across_several_surrogates_is_updated_in_all(domain):
    """Duplication scales with outputs -- one surrogate per output."""
    two = Domain(
        inputs=domain.inputs,
        outputs=Outputs(
            features=[
                ContinuousOutput(key="yield", objective=MaximizeObjective()),
                ContinuousOutput(key="cost", objective=MinimizeObjective()),
            ]
        ),
    )
    spec = MoboStrategy(domain=two).model_dump(mode="json")
    result = reconfigure(spec, NARROW, "narrow")
    assert len(result["propagated"]) == 2
    for output in ("yield", "cost"):
        assert bounds_at(
            result["spec"], f"surrogate_specs.{output}.inputs.temperature.bounds"
        ) == [60.0, 90.0]


def test_stepwise_substrategies_are_resynced(domain):
    """The silent case.

    `validate_domain_compatibility` ignores bounds and compares only the key
    intersection, so a stale step domain validates cleanly and optimises the
    wrong problem. Nothing but this pass catches it.
    """
    spec = StepwiseStrategy(
        domain=domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=domain),
                condition=NumberOfExperimentsCondition(n_experiments=2),
            ),
            Step(
                strategy_data=SoboStrategy(domain=domain),
                condition=AlwaysTrueCondition(),
            ),
        ],
    ).model_dump(mode="json")

    result = reconfigure(spec, NARROW, "narrow")
    for path in (
        "steps[0].strategy_data.domain.inputs.temperature.bounds",
        "steps[1].strategy_data.domain.inputs.temperature.bounds",
        "steps[1].strategy_data.surrogate_specs.yield.inputs.temperature.bounds",
    ):
        assert bounds_at(result["spec"], path) == [60.0, 90.0], path


def test_deliberately_diverged_copies_are_left_alone(domain):
    """A step may legitimately search a narrower region; only in-sync copies track."""
    narrow_domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(100, 120), unit="degC"),
                ContinuousInput(key="pressure", bounds=(1, 10), unit="bar"),
            ]
        ),
        outputs=domain.outputs,
    )
    spec = StepwiseStrategy(
        domain=domain,
        steps=[
            Step(
                strategy_data=RandomStrategy(domain=narrow_domain),
                condition=NumberOfExperimentsCondition(n_experiments=2),
            ),
            Step(
                strategy_data=RandomStrategy(domain=domain),
                condition=AlwaysTrueCondition(),
            ),
        ],
    ).model_dump(mode="json")

    result = reconfigure(spec, NARROW, "narrow")
    assert bounds_at(
        result["spec"], "steps[0].strategy_data.domain.inputs.temperature.bounds"
    ) == [100.0, 120.0]
    assert bounds_at(
        result["spec"], "steps[1].strategy_data.domain.inputs.temperature.bounds"
    ) == [60.0, 90.0]


def _with_extra_input(spec):
    features = spec["domain"]["inputs"]["features"] + [
        ContinuousInput(key="catalyst", bounds=(0.1, 5)).model_dump(mode="json")
    ]
    return [{"path": "domain.inputs.features", "value": features}]


def test_structural_change_regenerates_an_auto_derived_block(spec):
    result = reconfigure(spec, _with_extra_input(spec), "add catalyst")
    assert result["reset"] == ["surrogate_specs"]
    assert sorted(
        describe("surrogate_specs.yield.inputs", instance=result["spec"])["keys"]
    ) == [
        "catalyst",
        "pressure",
        "temperature",
    ]


def test_structural_change_refuses_to_discard_customised_config(custom_spec):
    """Regenerating would silently throw away the user's kernel choice."""
    with pytest.raises(ResyncError, match="surrogate_specs"):
        reconfigure(custom_spec, _with_extra_input(custom_spec), "add catalyst")


def test_refusal_leaves_the_input_untouched(custom_spec):
    import copy

    before = copy.deepcopy(custom_spec)
    with pytest.raises(ResyncError):
        reconfigure(custom_spec, _with_extra_input(custom_spec), "add catalyst")
    assert custom_spec == before


def test_removing_an_input_also_regenerates(spec):
    remaining = spec["domain"]["inputs"]["features"][:1]
    result = reconfigure(
        spec,
        [
            {"path": "domain.inputs.features", "value": remaining},
            {"path": "domain.constraints", "value": []},
        ],
        "drop pressure",
    )
    assert describe("surrogate_specs.yield.inputs", instance=result["spec"])[
        "keys"
    ] == {"temperature": "ContinuousInput"}


def test_no_resync_when_the_domain_is_untouched(spec):
    result = reconfigure(spec, [{"path": "folds", "value": 7}], "unrelated")
    assert result["propagated"] == []
    assert result["reset"] == []
