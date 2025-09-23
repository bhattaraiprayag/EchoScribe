# backend/config_manager.py

import logging
import os
from typing import Any, Dict

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yaml")


logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration from the file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {CONFIG_PATH}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return {}


def save_config(config: Dict[str, Any]):
    """Saves the configuration to the YAML file."""
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Configuration saved successfully.")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


config_data = load_config()
