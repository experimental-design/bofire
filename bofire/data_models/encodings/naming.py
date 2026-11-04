"""Naming convention for encoded feature columns.

Kept as a dependency-free leaf module so the encoding classes can import it at
module level without an import cycle back through ``features``.
"""


def get_encoded_name(feature_key: str, option_name: str) -> str:
    """Name of an encoded column: ``{feature_key}_{option_name}``.

    ``option_name`` is the category (one-hot/dummy) or descriptor name.
    """
    return f"{feature_key}_{option_name}"
