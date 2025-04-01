from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
from abc import ABC, abstractmethod
import inspect

from assistant.core.config_manager import ConfigManager
from assistant.utils.utils import title_to_snake


class Component(ABC):
    def __init__(
        self, name: Optional[str] = None, config: Optional[ConfigManager] = None
    ):
        self._name = name or title_to_snake(self.__class__.__name__)

        self.config = config.get_plugin_config(self.name) if config else {}
        self.logger = logging.getLogger(f"component.{name}")
        self.logger.setLevel(logging.INFO)
        self.event_handlers: Dict[str, List[Callable]] = {}

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def events(self) -> List[str]:
        pass

    def on(self, event: str, callback: Callable):
        if event not in self.events:
            raise ValueError(
                f"Component '{self.name}' does not produce '{event}' event."
            )
        if event not in self.event_handlers:
            self.event_handlers[event] = []

        self.event_handlers[event].append(callback)

    def proxy(self, event: str) -> Callable:
        def wrapper(*args, **kwargs):
            if event in self.event_handlers:
                for handler in self.event_handlers[event]:
                    handler(*args, **kwargs)
            else:
                self.logger.warning(f"No event handler for '{event}' event.")

        return wrapper

    def get_services(self) -> List[Tuple[str, Callable]]:
        services = []
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if hasattr(method, "_is_service"):
                service_name = getattr(method, "_service_name", name)
                services.append((service_name, method))
                self.logger.debug(
                    f"Found and registered decorated service: {service_name}"
                )

        return services

    def initialize(self) -> None:
        self.logger.info(f"Initializing {self._name}({self.version})")

    def shutdown(self) -> None:
        self.logger.info(f"Shutting down '{self._name}'")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value for this plugin."""
        return self.config.get(key, default)
