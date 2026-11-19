from bofire.data_models.migration.registry import normalizer


STEP = "pre_to_0_3_3"


@normalizer(STEP, "ContinuousOutput")
def normalize_continuous_output(p: dict) -> dict:
    # Some legacy RandomStrategy records carry `"objective": {}`, which is not
    # a valid AnyObjective discriminated-union member. ContinuousOutput.objective
    # is Optional, so collapse the empty dict to None.
    if p.get("objective") == {}:
        p["objective"] = None
    return p


@normalizer(STEP, "MaximizeObjective", "MinimizeObjective", "IdentityObjective")
def normalize_identity_objective(p: dict) -> dict:
    """Collapse legacy ``lower_bound`` / ``upper_bound`` fields into ``bounds``.

    The current ``IdentityObjective`` keeps ``bounds: [low, high]`` as a single
    field; ``lower_bound`` and ``upper_bound`` are now read-only properties.
    """
    has_lb = "lower_bound" in p
    has_ub = "upper_bound" in p
    if has_lb or has_ub:
        lb = p.pop("lower_bound", 0.0)
        ub = p.pop("upper_bound", 1.0)
        p.setdefault("bounds", [lb, ub])
    return p


@normalizer(STEP, "CategoricalInput", "CategoricalDescriptorInput")
def normalize_categorical_input(p: dict) -> dict:
    cats = p.get("categories")
    if isinstance(cats, list):
        p["categories"] = [str(c) for c in cats]
    return p


@normalizer(STEP, "Domain")
def normalize_domain(p: dict) -> dict:
    """Cross-cutting Domain fixups that need access to both inputs and constraints.

    Specifically: for every ``ContinuousInput`` that appears in a
    ``NChooseKConstraint`` with ``bounds[0] > 0`` and ``allow_zero`` not set,
    we set ``allow_zero=True``. The legacy schema didn't require this; the
    current NChooseK validator does.
    """
    constraints = p.get("constraints")
    if isinstance(constraints, dict):
        cs = constraints.get("constraints", []) or []
    elif isinstance(constraints, list):
        cs = constraints
    else:
        cs = []
    nchoosek_feature_keys: set = set()
    for c in cs:
        if isinstance(c, dict) and c.get("type") == "NChooseKConstraint":
            for k in c.get("features", []) or []:
                nchoosek_feature_keys.add(k)
    if not nchoosek_feature_keys:
        return p
    inputs = p.get("inputs")
    if isinstance(inputs, dict):
        feats = inputs.get("features", []) or []
    elif isinstance(inputs, list):
        feats = inputs
    else:
        feats = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        if f.get("type") != "ContinuousInput":
            continue
        if f.get("key") not in nchoosek_feature_keys:
            continue
        bounds = f.get("bounds")
        if not (isinstance(bounds, list) and len(bounds) == 2):
            continue
        if bounds[0] > 0 and not f.get("allow_zero"):
            f["allow_zero"] = True
    return p
