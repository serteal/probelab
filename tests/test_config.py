"""Tests for probelab config module."""

import os
from unittest.mock import patch

import pytest

from probelab.config import (
    ConfigVar,
    Context,
    DEBUG,
    DEFAULT_DEVICE,
    DISABLE_PROGRESS,
    LOG_LEVEL,
    PROFILE,
    VERBOSE,
    _bool_converter,
    _int_converter,
    get_config,
    set_defaults,
)


class TestBoolConverter:
    """Tests for _bool_converter function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("on", True),
            ("ON", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
            ("random", False),
        ],
    )
    def test_bool_converter(self, value: str, expected: bool):
        assert _bool_converter(value) == expected


class TestIntConverter:
    """Tests for _int_converter function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0", 0),
            ("1", 1),
            ("42", 42),
            ("-1", -1),
            ("100", 100),
        ],
    )
    def test_int_converter(self, value: str, expected: int):
        assert _int_converter(value) == expected

    def test_int_converter_invalid(self):
        with pytest.raises(ValueError):
            _int_converter("not_a_number")


class TestConfigVar:
    """Tests for ConfigVar class."""

    def test_default_value(self):
        var = ConfigVar("TEST_VAR", "default_value")
        assert var.get() == "default_value"

    def test_set_and_get(self):
        var = ConfigVar("TEST_VAR2", "default")
        token = var.set("new_value")
        assert var.get() == "new_value"
        var.reset(token)
        assert var.get() == "default"

    def test_env_var_fallback_string(self):
        with patch.dict(os.environ, {"PROBELAB_MY_TEST": "env_value"}):
            var = ConfigVar("MY_TEST", "default")
            # Note: ConfigVar reads env at creation time, so we need to create it within the patch
            assert var.get() == "env_value"

    def test_env_var_with_converter(self):
        with patch.dict(os.environ, {"PROBELAB_MY_BOOL": "true"}):
            var = ConfigVar("MY_BOOL", False, _bool_converter)
            assert var.get() is True

    def test_env_var_int_converter(self):
        with patch.dict(os.environ, {"PROBELAB_MY_INT": "42"}):
            var = ConfigVar("MY_INT", 0, _int_converter)
            assert var.get() == 42

    def test_repr(self):
        var = ConfigVar("TEST_REPR", "value")
        assert "TEST_REPR" in repr(var)
        assert "value" in repr(var)


class TestContext:
    """Tests for Context context manager."""

    def test_single_override(self):
        original = DEBUG.get()
        with Context(DEBUG=4):
            assert DEBUG.get() == 4
        assert DEBUG.get() == original

    def test_multiple_overrides(self):
        original_debug = DEBUG.get()
        original_verbose = VERBOSE.get()

        with Context(DEBUG=5, VERBOSE=False):
            assert DEBUG.get() == 5
            assert VERBOSE.get() is False

        assert DEBUG.get() == original_debug
        assert VERBOSE.get() == original_verbose

    def test_nested_contexts(self):
        original = DEBUG.get()

        with Context(DEBUG=1):
            assert DEBUG.get() == 1
            with Context(DEBUG=2):
                assert DEBUG.get() == 2
            assert DEBUG.get() == 1

        assert DEBUG.get() == original

    def test_case_insensitive_keys(self):
        original = DEBUG.get()

        with Context(debug=3):  # lowercase
            assert DEBUG.get() == 3

        with Context(Debug=4):  # mixed case
            assert DEBUG.get() == 4

        assert DEBUG.get() == original

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            with Context(UNKNOWN_KEY="value"):
                pass

    def test_context_returns_self(self):
        with Context(DEBUG=1) as ctx:
            assert isinstance(ctx, Context)


class TestGetConfig:
    """Tests for get_config function."""

    def test_returns_dict(self):
        config = get_config()
        assert isinstance(config, dict)

    def test_contains_all_vars(self):
        config = get_config()
        assert "LOG_LEVEL" in config
        assert "VERBOSE" in config
        assert "DEFAULT_DEVICE" in config
        assert "DISABLE_PROGRESS" in config
        assert "DEBUG" in config
        assert "PROFILE" in config

    def test_reflects_context_changes(self):
        with Context(DEBUG=99):
            config = get_config()
            assert config["DEBUG"] == 99


class TestSetDefaults:
    """Tests for set_defaults function."""

    def test_set_single_default(self):
        original = DEBUG.get()
        try:
            set_defaults(DEBUG=10)
            assert DEBUG.get() == 10
        finally:
            # Restore original
            set_defaults(DEBUG=original)

    def test_set_multiple_defaults(self):
        original_debug = DEBUG.get()
        original_verbose = VERBOSE.get()
        try:
            set_defaults(DEBUG=20, VERBOSE=False)
            assert DEBUG.get() == 20
            assert VERBOSE.get() is False
        finally:
            set_defaults(DEBUG=original_debug, VERBOSE=original_verbose)

    def test_case_insensitive(self):
        original = DEBUG.get()
        try:
            set_defaults(debug=15)  # lowercase
            assert DEBUG.get() == 15
        finally:
            set_defaults(DEBUG=original)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            set_defaults(UNKNOWN_VAR="value")


class TestDefaultValues:
    """Tests that default values are sensible."""

    def test_log_level_default(self):
        # Should be a valid log level by default
        assert LOG_LEVEL.get() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_verbose_default_is_bool(self):
        assert isinstance(VERBOSE.get(), bool)

    def test_default_device_is_string(self):
        assert isinstance(DEFAULT_DEVICE.get(), str)
        assert DEFAULT_DEVICE.get() in ("cuda", "cpu", "mps") or DEFAULT_DEVICE.get().startswith("cuda:")

    def test_disable_progress_default_is_bool(self):
        assert isinstance(DISABLE_PROGRESS.get(), bool)

    def test_debug_default_is_int(self):
        assert isinstance(DEBUG.get(), int)

    def test_profile_default_is_bool(self):
        assert isinstance(PROFILE.get(), bool)


class TestIntegrationWithProbelab:
    """Integration tests with probelab imports."""

    def test_import_from_probelab(self):
        import probelab as pl

        # Context is available at top level
        assert hasattr(pl, "Context")

        # get_config and set_defaults require explicit import
        from probelab.config import get_config, set_defaults
        assert callable(get_config)
        assert callable(set_defaults)

    def test_context_works_from_top_level(self):
        import probelab as pl

        original = DEBUG.get()
        with pl.Context(DEBUG=42):
            assert DEBUG.get() == 42
        assert DEBUG.get() == original
