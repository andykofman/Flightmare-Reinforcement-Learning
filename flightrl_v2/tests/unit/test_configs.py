"""Unit tests for config module."""
import pytest
from pathlib import Path
import tempfile


class TestConfigLoader:
    """Tests for config loading utilities."""

    def test_load_config(self):
        """Test loading a YAML config."""
        from flightrl_v2.configs import load_config

        # Create temp config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test_key: test_value\n")
            f.write("nested:\n  inner: 42\n")
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["test_key"] == "test_value"
            assert config["nested"]["inner"] == 42
        finally:
            Path(temp_path).unlink()

    def test_config_overrides(self):
        """Test config with overrides."""
        from flightrl_v2.configs import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key1: value1\n")
            f.write("key2: value2\n")
            temp_path = f.name

        try:
            config = load_config(temp_path, overrides={"key1": "overridden"})
            assert config["key1"] == "overridden"
            assert config["key2"] == "value2"
        finally:
            Path(temp_path).unlink()

    def test_save_config(self):
        """Test saving a config to YAML."""
        from flightrl_v2.configs import save_config, load_config

        test_config = {
            "test_key": "test_value",
            "nested": {"inner": 42}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_config(test_config, temp_path)
            loaded_config = load_config(temp_path)
            assert loaded_config["test_key"] == "test_value"
            assert loaded_config["nested"]["inner"] == 42
        finally:
            Path(temp_path).unlink()

    def test_validate_config(self):
        """Test config validation."""
        from flightrl_v2.configs import validate_config
        from flightrl_v2.configs.schema import ConfigValidationError

        # Valid config
        config = {"key1": "value", "key2": 42}
        assert validate_config(config, required_keys=["key1"]) is True

        # Missing required key
        with pytest.raises(ConfigValidationError):
            validate_config(config, required_keys=["missing_key"])

    def test_default_config_path(self):
        """Test get_default_config_path function."""
        from flightrl_v2.configs.loader import get_default_config_path

        path = get_default_config_path("hover")
        assert path.name == "hover.yaml"
        assert "defaults" in str(path)

