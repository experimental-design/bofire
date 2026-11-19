from typing import Iterable

from bofire.data_models.migration.errors import UnrecoverablePayloadError


def drop_keys(d: dict, keys: Iterable[str]) -> dict:
    for k in keys:
        d.pop(k, None)
    return d


_LEGACY_SCALER_TO_OBJECT = {
    "NORMALIZE": {"type": "Normalize", "features": []},
    "STANDARDIZE": {"type": "Standardize", "features": []},
    "IDENTITY": None,
}

_SCALER_UNRECOVERABLE_ON_INPUT_SIDE = {"LOG", "CHAINED_LOG_STANDARDIZE"}


def normalize_scaler_field(d: dict, key: str = "scaler") -> dict:
    """Convert a legacy string scaler value to the current object form.

    Only the input-side ``scaler`` field changed (string enum -> tagged-union
    object). ``output_scaler`` remains a ``ScalerEnum`` string, so this atom
    must not be called on ``output_scaler``.
    """
    if key not in d:
        return d
    value = d[key]
    if isinstance(value, str):
        if value in _SCALER_UNRECOVERABLE_ON_INPUT_SIDE:
            raise UnrecoverablePayloadError(
                payload_type=d.get("type", "<unknown>"),
                reason=(
                    f"scaler={value!r} is no longer valid for the input scaler; "
                    f"only Normalize/Standardize/None are accepted."
                ),
                hint=(
                    "Re-fit the surrogate with scaler=Normalize() (or None) and "
                    "use output_scaler for log transforms."
                ),
            )
        if value not in _LEGACY_SCALER_TO_OBJECT:
            raise UnrecoverablePayloadError(
                payload_type=d.get("type", "<unknown>"),
                reason=f"Unknown legacy scaler value {value!r}.",
            )
        d[key] = _LEGACY_SCALER_TO_OBJECT[value]
    return d


_HYPERSTRATEGY_RENAMES = {
    "FactorialStrategy": "FractionalFactorialStrategy",
}


def normalize_hyperstrategy(hc: dict) -> dict:
    """Apply known renames to ``hyperstrategy`` in a hyperconfig dict."""
    if not isinstance(hc, dict):
        return hc
    name = hc.get("hyperstrategy")
    if name in _HYPERSTRATEGY_RENAMES:
        hc["hyperstrategy"] = _HYPERSTRATEGY_RENAMES[name]
    return hc


_CATEGORICAL_ENCODING_STRINGS = {"ONE_HOT", "ORDINAL", "DUMMY", "DESCRIPTOR"}


def split_input_preprocessing_specs(p: dict) -> dict:
    """Move legacy categorical encodings into ``categorical_encodings``.

    Legacy BotorchSurrogate payloads stored categorical encodings (``ONE_HOT``,
    ``DESCRIPTOR``, etc.) in ``input_preprocessing_specs``. The current schema
    forces ``ORDINAL`` for categoricals in ``input_preprocessing_specs`` and
    carries the "within-the-model" encoding on ``categorical_encodings``.

    String values are categorical encoding enums and get moved. Dict values
    (``Fingerprints`` etc. for molecular features) stay put; the
    ``categorical_encodings`` validator routes them based on feature type.
    """
    ips = p.get("input_preprocessing_specs")
    if not isinstance(ips, dict) or not ips:
        return p
    ce = p.setdefault("categorical_encodings", {})
    remaining = {}
    for key, value in ips.items():
        if isinstance(value, str) and value in _CATEGORICAL_ENCODING_STRINGS:
            ce.setdefault(key, value)
        else:
            remaining[key] = value
    p["input_preprocessing_specs"] = remaining
    return p
