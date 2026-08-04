"""Tests for bofire.llm.mapper."""

import os

import pytest

from bofire.llm.mapper import _resolve_env_var


def test_resolve_env_var_success():
    os.environ["_BOFIRE_TEST_VAR"] = "test_value"
    try:
        assert _resolve_env_var("_BOFIRE_TEST_VAR") == "test_value"
    finally:
        del os.environ["_BOFIRE_TEST_VAR"]


def test_resolve_env_var_missing():
    with pytest.raises(EnvironmentError, match="NONEXISTENT_VAR_12345"):
        _resolve_env_var("NONEXISTENT_VAR_12345")
