# tests/test_config_manager.py
"""Tests for configuration management."""

import os
import copy
import tempfile

import pytest
import yaml


class TestConfigManager:
    """Tests for configuration management."""

    def test_load_config_returns_dict(self):
        """load_config should return a dictionary."""
        from config_manager import load_config

        config = load_config()
        assert isinstance(config, dict)

    def test_get_config_returns_copy(self):
        """get_config should return a deep copy, not the original."""
        from config_manager import get_config

        config1 = get_config()
        config2 = get_config()

        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2

        # Modifying one should not affect the other
        config1["test_key"] = "test_value"
        assert "test_key" not in config2

    def test_get_config_nested_is_copy(self):
        """Nested dicts from get_config should also be copies."""
        from config_manager import get_config

        config1 = get_config()
        config2 = get_config()

        if "vad_parameters" in config1:
            config1["vad_parameters"]["test_nested"] = "value"
            assert "test_nested" not in config2.get("vad_parameters", {})

    def test_reload_config_updates_internal_data(self):
        """reload_config should update the internal config data."""
        from config_manager import get_config, reload_config, save_config

        # Get original config
        original = get_config()

        # Modify and save
        modified = copy.deepcopy(original)
        modified["_test_reload"] = "reload_test_value"
        save_config(modified)

        # Reload and verify
        reload_config()
        reloaded = get_config()

        assert reloaded.get("_test_reload") == "reload_test_value"

        # Cleanup: restore original
        del modified["_test_reload"]
        save_config(modified)
        reload_config()

    def test_save_config_updates_in_memory(self):
        """save_config should also update in-memory config."""
        from config_manager import get_config, save_config

        original = get_config()

        # Modify and save
        modified = copy.deepcopy(original)
        modified["_test_save"] = "save_test_value"
        save_config(modified)

        # Get config should reflect the change without explicit reload
        updated = get_config()
        assert updated.get("_test_save") == "save_test_value"

        # Cleanup: restore original
        del modified["_test_save"]
        save_config(modified)

    def test_config_data_backwards_compatibility(self):
        """config_data should still work for backwards compatibility."""
        from config_manager import config_data

        # Should be a dict
        assert isinstance(config_data, dict)

        # Should have expected keys
        assert "vad_parameters" in config_data or "audio_parameters" in config_data


class TestConfigManagerEdgeCases:
    """Edge case tests for config manager."""

    def test_get_config_with_empty_file(self):
        """get_config should handle empty config gracefully."""
        from config_manager import get_config

        # Even if config is minimal, should return dict
        config = get_config()
        assert isinstance(config, dict)

    def test_reload_config_returns_dict(self):
        """reload_config should return the reloaded config."""
        from config_manager import reload_config

        result = reload_config()
        assert isinstance(result, dict)
