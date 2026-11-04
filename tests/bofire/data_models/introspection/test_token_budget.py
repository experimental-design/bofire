"""Budget regression gate.

The whole point of this package is cost. These budgets are byte-based -- the
``testing_data_models`` CI job installs only the package and pytest, so no
tokenizer is available -- and carry roughly 30% headroom.

The budgets are *expected to rise* as the documentation ratchet reaches
strategies, surrogates and acquisition functions: every ``Field(description=...)``
that lands makes the summary more useful and slightly larger. Raise them
deliberately when that happens; do not raise them to accommodate a regression in
how much a node expands.
"""

import json

import pytest

from bofire.data_models.introspection.api import approx_tokens, describe, full_schema
from bofire.data_models.introspection.registry import lookup


#: The three-call walk from the feature request, per call.
NAVIGATION_BUDGET = {
    "SoboStrategy": 400,
    "SoboStrategy.acquisition_function": 80,
    "qUCB": 250,
}

#: Nodes whose naive schema is catastrophic but whose shallow form is not.
SHALLOW_BUDGET = {
    "SingleTaskGPSurrogate": 1400,
    "BotorchSurrogates": 700,
    "Domain": 500,
}


@pytest.mark.parametrize("path, budget", sorted(NAVIGATION_BUDGET.items()))
def test_each_navigation_call_stays_within_budget(path, budget):
    assert approx_tokens(describe(path.split("."))) < budget


def test_the_whole_walk_is_orders_of_magnitude_cheaper_than_the_raw_schema():
    walk = sum(approx_tokens(describe(path.split("."))) for path in NAVIGATION_BUDGET)
    naive = approx_tokens(lookup("SoboStrategy").model_json_schema())
    assert walk < 800
    assert naive / walk > 50


@pytest.mark.parametrize("name, budget", sorted(SHALLOW_BUDGET.items()))
def test_full_schema_is_bounded_by_the_node_not_the_graph(name, budget):
    """Guards the ``$defs``-stripping in ``shallow_schema``.

    The second assertion is the important one: it pins that the naive
    alternative is still catastrophic, so replacing ``shallow_schema`` with a
    plain ``model_json_schema()`` call fails loudly instead of silently
    reintroducing a 25k-token response.
    """
    assert approx_tokens(full_schema(name)) < budget
    assert len(json.dumps(lookup(name).model_json_schema())) > 40_000


def test_omitted_keys_are_never_emitted_as_null():
    """Absent descriptions must cost nothing; most data models are undocumented."""

    def walk(payload):
        if isinstance(payload, dict):
            assert None not in payload.values(), payload
            for value in payload.values():
                walk(value)
        elif isinstance(payload, list):
            for item in payload:
                walk(item)

    walk(describe(["qUCB"]))
    walk(describe(["SoboStrategy", "acquisition_function"]))
