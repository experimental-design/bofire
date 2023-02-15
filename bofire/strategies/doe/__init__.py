import warnings

try:
    import cyipopt  # noqa: F401
except ImportError as e:
    warnings.warn(e.msg)
    warnings.warn(
        "please run `conda install -c conda-forge cyipopt` for this functionality."
    )
