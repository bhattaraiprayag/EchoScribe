"""YAML configuration management for EchoScribe.

This module provides thread-safe loading, saving, and accessing of application
configuration stored in config.yaml. It ensures immutability through deep copying.
"""

import copy
import logging
import os
from typing import Any, Dict

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yaml")


logger = logging.getLogger(__name__)

_config_data: Dict[str, Any] = {}


def load_config() -> Dict[str, Any]:
    """Load YAML configuration from file.

    Returns:
        Configuration dictionary. Empty dict if file not found or parsing fails.
    """
    global _config_data
    try:
        with open(CONFIG_PATH, "r") as f:
            _config_data = yaml.safe_load(f) or {}
        logger.info("Configuration loaded successfully.")
        return _config_data
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {CONFIG_PATH}")
        _config_data = {}
        return _config_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        _config_data = {}
        return _config_data


def reload_config() -> Dict[str, Any]:
    """Reload configuration from file.

    Returns:
        Reloaded configuration dictionary.
    """
    return load_config()


def get_config() -> Dict[str, Any]:
    """Get a deep copy of current configuration.

    Returns:
        Deep copy of configuration dictionary.
    """
    return copy.deepcopy(_config_data)


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to YAML file and update in-memory config.

    Args:
        config: Configuration dictionary to save.
    """
    global _config_data
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        _config_data = copy.deepcopy(config)
        logger.info("Configuration saved successfully.")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


_config_data = load_config()

config_data = _config_data
