import os
import importlib
import inspect
import pkgutil
import logging
from typing import Dict, List, Optional, Tuple
from .event_bus import EventBus
from .plugin import Plugin

class PluginManager:
    def __init__(self, plugin_dirs=None):
        self.event_bus = EventBus()
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def discover_plugins(self) -> List[Tuple[str, type]]:
        """Discover all plugin classes in the plugin directories."""
        plugin_classes = []
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                self.logger.warning(f"Plugin directory '{plugin_dir}' does not exist")
                continue

            for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                if is_pkg:
                    try:
                        package_name = f"{plugin_dir}.{name}"
                        module_name = f"{package_name}.plugin"

                        try:
                            module = importlib.import_module(module_name)
                        except ImportError:
                            self.logger.debug(f"No plugin.py in '{package_name}', trying __init__")
                            module = importlib.import_module(package_name)

                        for _, cls in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(cls, Plugin) and
                                cls is not Plugin and
                                cls.__module__ == module.__name__):
                                plugin_classes.append((name, cls))
                                self.logger.debug(f"Discovered plugin: {name}('{cls.__name__}')")
                    except Exception as e:
                        self.logger.error(f"Error loading plugin '{name}': {e}")

        return plugin_classes

    def load_plugins(self) -> None:
        """Load all discovered plugins."""
        plugin_classes = self.discover_plugins()
        for name, plugin_class in plugin_classes:
            try:
                plugin = plugin_class(name, self.event_bus)
                self.plugins[name] = plugin
                self.logger.info(f"Loaded plugin: {name}({plugin.version})")
            except Exception as e:
                self.logger.error(f"Error instantiating plugin '{name}': {e}")

    def register_plugin_events(self) -> None:
        """Register events for all loaded plugins."""
        for name, plugin in self.plugins.items():
            try:
                if plugin.enabled:
                    plugin.register_events()
            except Exception as e:
                self.logger.error(f"Error registering events for plugin '{name}': {e}")
                plugin.disable()

    def resolve_dependencies(self) -> List[str]:
        """Resolve plugin dependencies and return a sorted list of plugin names for initialization."""
        # Check for missing dependencies
        for name, plugin in self.plugins.items():
            for dep in plugin.required_plugins:
                if dep not in self.plugins:
                    self.logger.error(f"Plugin '{name}' depends on '{dep}', but it's not loaded")
                    plugin.disable()

        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []

        def visit(name):
            if name in temp_visited:
                self.logger.error(f"Circular dependency detected involving '{name}'")
                return False
            if name in visited:
                return True

            temp_visited.add(name)

            plugin = self.plugins.get(name)
            if plugin:
                for dep in plugin.required_plugins:
                    if dep in self.plugins and not visit(dep):
                        return False

            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
            return True

        # Visit all plugins
        for name in list(self.plugins.keys()):
            if name not in visited:
                if not visit(name):
                    # Circular dependency, disable the plugin
                    self.plugins[name].disable()

        return order

    def initialize_plugins(self) -> None:
        """Initialize all plugins in dependency order."""
        # First register all events
        self.register_plugin_events()

        # Then initialize in dependency order
        init_order = self.resolve_dependencies()

        for name in init_order:
            plugin = self.plugins.get(name)
            if plugin and plugin.enabled:
                try:
                    plugin.initialize()
                except Exception as e:
                    self.logger.error(f"Error initializing plugin '{name}': {e}")
                    plugin.disable()

    def shutdown_plugins(self) -> None:
        """Shutdown all plugins in reverse dependency order."""
        shutdown_order = self.resolve_dependencies()
        shutdown_order.reverse()

        for name in shutdown_order:
            plugin = self.plugins.get(name)
            if plugin:
                try:
                    plugin.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down plugin '{name}': {e}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)

