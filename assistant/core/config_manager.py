import os
import yaml
from typing import Dict, Any


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            return {"system": {}, "plugins": {}}

        with open(self.config_path, "r") as file:
            return yaml.safe_load(file) or {"system": {}, "plugins": {}}

    def get_system_config(self) -> Dict[str, Any]:
        """Get system-wide configuration."""
        return self.config.get("system", {})

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        plugins_config = self.config.get("plugins", {})
        return plugins_config.get(plugin_name, {})

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled in the configuration."""
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config.get("enabled", False)
