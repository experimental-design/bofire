from bofire.data_models.migration.registry import normalizer
from bofire.data_models.migration.walker import RECURSE_MAP, walk


STEP = "_test_walker_step"
ORDER: list = []


def _register(tag: str, recurse_into=None):
    if recurse_into is not None:
        RECURSE_MAP[tag] = recurse_into

    def fn(p: dict) -> dict:
        ORDER.append(p["type"])
        return p

    normalizer(STEP, tag)(fn)


def setup_function():
    ORDER.clear()


def test_bottom_up_order():
    _register("__TestParent", {"child": "typed"})
    _register("__TestChild", {})

    walk({"type": "__TestParent", "child": {"type": "__TestChild"}}, STEP)
    assert ORDER == ["__TestChild", "__TestParent"]


def test_typed_or_null_skips_none():
    _register("__TestRoot", {"opt": "typed_or_null"})
    walk({"type": "__TestRoot", "opt": None}, STEP)
    assert ORDER == ["__TestRoot"]


def test_list_of_typed_walks_each_child():
    _register("__TestList", {"items": "list_of_typed"})
    _register("__TestItem", {})
    walk(
        {
            "type": "__TestList",
            "items": [{"type": "__TestItem"}, {"type": "__TestItem"}],
        },
        STEP,
    )
    assert ORDER.count("__TestItem") == 2
    assert ORDER[-1] == "__TestList"


def test_inferred_type_for_known_structural_key():
    _register("__TestDomainLike", {"inputs": "container"})
    payload = {"type": "__TestDomainLike", "inputs": {"features": []}}
    walk(payload, STEP)
    # Inputs container was untagged; walker infers "Inputs" before recursing.
    assert payload["inputs"]["type"] == "Inputs"


def test_tagless_container_recurses_via_structural_map():
    # BotorchSurrogates is a tagless container; walker uses
    # STRUCTURAL_RECURSE_BY_KEY to descend into its `surrogates` list.
    _register("__TestStratWithSpecs", {"surrogate_specs": "typed_or_null"})
    _register("__TestInnerSurrogate", {})
    payload = {
        "type": "__TestStratWithSpecs",
        "surrogate_specs": {"surrogates": [{"type": "__TestInnerSurrogate"}]},
    }
    walk(payload, STEP)
    assert "__TestInnerSurrogate" in ORDER
    # The tagless container should NOT have been written with a `type` key.
    assert "type" not in payload["surrogate_specs"]
