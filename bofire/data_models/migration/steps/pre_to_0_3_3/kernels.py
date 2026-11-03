from bofire.data_models.migration.registry import normalizer
from bofire.data_models.migration.steps.pre_to_0_3_3._atoms import (
    normalize_hyperstrategy,
)


STEP = "pre_to_0_3_3"


# Hyperconfigs need the legacy "FactorialStrategy" -> "FractionalFactorialStrategy"
# rename on the hyperstrategy field.
@normalizer(STEP, "SingleTaskGPHyperconfig", "MixedSingleTaskGPHyperconfig")
def normalize_hyperconfig(p: dict) -> dict:
    return normalize_hyperstrategy(p)


# A typo in legacy payloads — should always have been HammingDistanceKernel.
@normalizer(STEP, "HammondDistanceKernel")
def fix_hammond_typo(p: dict) -> dict:
    p["type"] = "HammingDistanceKernel"
    return p
