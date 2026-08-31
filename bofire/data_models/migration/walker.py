from typing import Dict, Literal

from bofire.data_models.migration.registry import get_normalizer


RecurseMode = Literal["typed", "typed_or_null", "list_of_typed", "container"]

RECURSE_MAP: Dict[str, Dict[str, RecurseMode]] = {
    "Domain": {
        "inputs": "container",
        "outputs": "container",
        "constraints": "container",
    },
    "Inputs": {"features": "list_of_typed"},
    "Outputs": {"features": "list_of_typed"},
    "Constraints": {"constraints": "list_of_typed"},
    "EngineeredFeatures": {"features": "list_of_typed"},
    "BotorchSurrogates": {"surrogates": "list_of_typed"},
    # Surrogate base recursion: most surrogates carry inputs/outputs and an
    # engineered_features container; the trainable botorch ones additionally
    # carry kernel(s), priors, scaler, hyperconfig. We register the union of
    # plausibly-present keys for each type; absent keys are skipped.
    "SingleTaskGPSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "kernel": "typed",
        "noise_prior": "typed",
        "noise_constraint": "typed_or_null",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "MixedSingleTaskGPSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "continuous_kernel": "typed",
        "categorical_kernel": "typed",
        "noise_prior": "typed",
        "noise_constraint": "typed_or_null",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "LinearSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "kernel": "typed",
        "noise_prior": "typed",
        "noise_constraint": "typed_or_null",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "FullyBayesianSingleTaskGPSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "AdditiveMapSaasSingleTaskGPSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "RandomForestSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "MLPEnsemble": {
        "inputs": "container",
        "outputs": "container",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "RegressionMLPEnsemble": {
        "inputs": "container",
        "outputs": "container",
        "hyperconfig": "typed_or_null",
        "scaler": "typed_or_null",
        "engineered_features": "typed",
    },
    "EmpiricalSurrogate": {
        "inputs": "container",
        "outputs": "container",
    },
    "LinearDeterministicSurrogate": {
        "inputs": "container",
        "outputs": "container",
        "engineered_features": "typed",
    },
    # Strategies
    "RandomStrategy": {"domain": "container"},
    "SoboStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
    },
    "MoboStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
        "ref_point": "typed_or_null",
    },
    "QnehviStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
    },
    "QparegoStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
    },
    "MultiplicativeSoboStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
    },
    "CustomSoboStrategy": {
        "domain": "container",
        "surrogate_specs": "typed_or_null",
        "acquisition_function": "typed_or_null",
        "acquisition_optimizer": "typed_or_null",
    },
    "FractionalFactorialStrategy": {"domain": "container"},
    "DoEStrategy": {"domain": "container", "criterion": "typed_or_null"},
    "StepwiseStrategy": {"domain": "container"},
    # Hyperconfig carries its own inputs container (hyperparameter search space).
    "SingleTaskGPHyperconfig": {"inputs": "container"},
    "MixedSingleTaskGPHyperconfig": {"inputs": "container"},
    # ContinuousOutput's objective field carries a typed AnyObjective dict.
    "ContinuousOutput": {"objective": "typed_or_null"},
    # Kernel structure: ScaleKernel wraps a base_kernel.
    "ScaleKernel": {"base_kernel": "typed", "outputscale_prior": "typed_or_null"},
    "MaternKernel": {"lengthscale_prior": "typed_or_null"},
    "RBFKernel": {"lengthscale_prior": "typed_or_null"},
    "LinearKernel": {"variance_prior": "typed_or_null"},
}


# Top-level type tags that may be missing in legacy payloads (the dump shows
# Inputs/Outputs/Constraints embedded without a "type" key). We patch those in
# before recursing so the walker can find a normalizer.
INFERRED_TYPE_KEYS: Dict[str, str] = {
    # Child structural key -> type tag to insert when the child dict is missing
    # its `type` field. Only used for classes that *have* a `type` Literal so
    # inserting it won't trigger extra="forbid".
    "inputs": "Inputs",
    "outputs": "Outputs",
    "constraints": "Constraints",
    "engineered_features": "EngineeredFeatures",
}


# Fallback recursion spec for tagless dicts encountered at a known parent
# structural key. The walker uses this when a child dict has no `type` field
# AND the child's class is not in INFERRED_TYPE_KEYS (e.g. `BotorchSurrogates`
# is a tagless container).
STRUCTURAL_RECURSE_BY_KEY: Dict[str, Dict[str, RecurseMode]] = {
    "surrogate_specs": {"surrogates": "list_of_typed"},
}


def walk(node, step: str, structural_key: str = ""):
    """Recursively normalize a payload bottom-up.

    Returns the normalized node. Mutates dicts in place but the return value
    is what callers should use (some normalizers replace the dict entirely).
    """
    if node is None:
        return None
    if isinstance(node, list):
        return [walk(child, step, structural_key) for child in node]
    if not isinstance(node, dict):
        return node

    type_tag = node.get("type")

    if type_tag and type_tag in RECURSE_MAP:
        spec = RECURSE_MAP[type_tag]
    elif type_tag is None and structural_key in STRUCTURAL_RECURSE_BY_KEY:
        spec = STRUCTURAL_RECURSE_BY_KEY[structural_key]
    else:
        spec = {}

    # Descend into known typed children first (bottom-up).
    for child_key, mode in spec.items():
        if child_key not in node:
            continue
        child = node[child_key]
        if mode == "typed":
            node[child_key] = walk(
                _with_inferred_type(child, child_key), step, child_key
            )
        elif mode == "typed_or_null":
            if child is None:
                continue
            node[child_key] = walk(
                _with_inferred_type(child, child_key), step, child_key
            )
        elif mode == "list_of_typed":
            if not isinstance(child, list):
                continue
            node[child_key] = [
                walk(_with_inferred_type(c, child_key), step, child_key) for c in child
            ]
        elif mode == "container":
            node[child_key] = walk(
                _with_inferred_type(child, child_key), step, child_key
            )

    # Now apply this node's normalizer.
    if type_tag is not None:
        normalizer_fn = get_normalizer(step, type_tag)
        node = normalizer_fn(node)

    return node


def _with_inferred_type(node, child_key: str):
    """If a child dict at a known structural key lacks a `type` tag, fill it in.

    Only applied to classes that have a Literal `type` field — never to tagless
    containers like ``BotorchSurrogates``.
    """
    if not isinstance(node, dict):
        return node
    if "type" in node:
        return node
    inferred = INFERRED_TYPE_KEYS.get(child_key)
    if inferred is not None:
        node["type"] = inferred
    return node
