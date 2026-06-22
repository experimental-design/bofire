from bofire.data_models.migration.errors import UnrecoverablePayloadError
from bofire.data_models.migration.registry import normalizer
from bofire.data_models.migration.steps.pre_to_0_3_3._atoms import (
    drop_keys,
    normalize_scaler_field,
    split_input_preprocessing_specs,
)


STEP = "pre_to_0_3_3"


def _normalize_trainable_botorch(p: dict) -> dict:
    drop_keys(p, ["aggregations"])
    normalize_scaler_field(p, "scaler")
    split_input_preprocessing_specs(p)
    return p


@normalizer(
    STEP,
    "SingleTaskGPSurrogate",
    "LinearSurrogate",
    "FullyBayesianSingleTaskGPSurrogate",
    "AdditiveMapSaasSingleTaskGPSurrogate",
    "RandomForestSurrogate",
)
def normalize_trainable_botorch(p: dict) -> dict:
    return _normalize_trainable_botorch(p)


@normalizer(STEP, "MixedSingleTaskGPSurrogate")
def normalize_mixed_single_task_gp(p: dict) -> dict:
    """MixedSingleTaskGPSurrogate requires ORDINAL encoding for categoricals.

    Some legacy payloads put ``ONE_HOT`` in ``input_preprocessing_specs`` for
    Mixed models. We drop the entry entirely so the BotorchSurrogate validator
    fills in ORDINAL automatically; ``categorical_encodings`` likewise defaults
    to ORDINAL for plain ``CategoricalInput`` on Mixed.
    """
    drop_keys(p, ["aggregations"])
    normalize_scaler_field(p, "scaler")
    ips = p.get("input_preprocessing_specs")
    if isinstance(ips, dict):
        # Strip string-valued (categorical-encoding) entries entirely; keep
        # molecular dict entries (Fingerprints etc.).
        p["input_preprocessing_specs"] = {
            k: v for k, v in ips.items() if not isinstance(v, str)
        }
    return p


@normalizer(STEP, "MLPEnsemble")
def migrate_mlp_ensemble(p: dict) -> dict:
    # Legacy "MLPEnsemble" type was renamed to "RegressionMLPEnsemble" when
    # the classification variant was introduced. The legacy class is the
    # regression model.
    p["type"] = "RegressionMLPEnsemble"
    return _normalize_trainable_botorch(p)


@normalizer(STEP, "RegressionMLPEnsemble")
def normalize_regression_mlp(p: dict) -> dict:
    return _normalize_trainable_botorch(p)


@normalizer(STEP, "EmpiricalSurrogate", "LinearDeterministicSurrogate")
def normalize_simple_surrogate(p: dict) -> dict:
    drop_keys(p, ["aggregations"])
    # EmpiricalSurrogate is a BotorchSurrogate too, so it has the same
    # input_preprocessing_specs / categorical_encodings split.
    split_input_preprocessing_specs(p)
    return p


@normalizer(STEP, "XGBoostSurrogate")
def reject_xgboost(p: dict) -> dict:
    raise UnrecoverablePayloadError(
        payload_type="XGBoostSurrogate",
        reason="XGBoostSurrogate was removed in BoFire 0.3.x.",
        hint="Re-fit the model as RandomForestSurrogate or LinearSurrogate.",
    )
