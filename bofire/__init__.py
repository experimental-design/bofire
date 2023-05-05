try:
    from bofire.version import version as __version__  # type: ignore
except Exception:
    __version__ = "Unknown"
