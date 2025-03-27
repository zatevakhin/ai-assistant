import logging
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Set, Callable, Optional, Tuple
from concurrent.futures import Future
from .event_bus import EventBus


class Plugin(ABC):
    def __init__(self, name: str, event_bus: EventBus, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{name}")
        self.logger.setLevel(logging.INFO)
        self.enabled = True
        self.dependencies = set()
        self.subscriptions = []
        self.registered_events = []
        self.registered_services = []

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin."""
        pass

    @abstractmethod
    def get_event_definitions(self) -> List[str]:
        """Return a list of event constants defined by this plugin."""
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value for this plugin."""
        return self.config.get(key, default)

    def register_events(self) -> None:
        """Register the plugin's events with the event bus."""
        event_ids = self.get_event_definitions()
        self.registered_events = self.event_bus.register_events(event_ids, self.name)
        if len(self.registered_events) != len(event_ids):
            failed = set(event_ids) - set(self.registered_events)
            self.logger.warning(f"Failed to register events: {', '.join(failed)}")
        else:
            self.logger.info(f"Registered {len(self.registered_events)} events")

    def add_dependency(self, plugin_name: str) -> None:
        """Add a plugin dependency."""
        self.dependencies.add(plugin_name)

    @property
    def required_plugins(self) -> Set[str]:
        """Return the set of required plugin names."""
        return self.dependencies

    def subscribe_to_event(self, event_id: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event with a callback function."""
        subscription = self.event_bus.subscribe(event_id, callback)
        if subscription:
            self.subscriptions.append(subscription)
            self.logger.debug(f"Subscribed to '{event_id}'")

    def publish_event(self, event_id: str, data: Any) -> None:
        """Publish an event with data."""
        if not self.enabled:
            return

        if event_id not in self.registered_events:
            self.logger.warning(f"Attempting to publish unregistered event: '{event_id}'")
            return

        self.event_bus.publish(event_id, data)

    def register_service(self, service_name: str, method: Callable) -> bool:
        """
        Register a service method with the event bus.
        """
        if not self.enabled:
            return False

        success = self.event_bus.register_service(self.name, service_name, method)
        if success:
            is_async = hasattr(method, "_is_async") and getattr(method, "_is_async")
            self.registered_services.append(service_name)
            self.logger.debug(f"Registered service '{service_name}' ({'async' if is_async else 'sync'})")
        return success

    def call_service(self, plugin_name: str, service_name: str, *args, **kwargs) -> Any:
        """
        Synchronously call a service method on another plugin.
        """
        if not self.enabled:
            raise RuntimeError(f"Plugin '{self.name}' is disabled")

        try:
            return self.event_bus.call_service(plugin_name, service_name, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error calling service '{service_name}' on plugin '{plugin_name}': {e}")
            raise

    def call_service_async(self, plugin_name: str, service_name: str, *args, **kwargs) -> Tuple[str, Future]:
        """
        Asynchronously call a service method on another plugin.
        """
        if not self.enabled:
            raise RuntimeError(f"Plugin '{self.name}' is disabled")

        try:
            return self.event_bus.call_service_async(plugin_name, service_name, *args, **kwargs)
        except ValueError as e:
            self.logger.error(f"Error starting async call to service '{service_name}' on plugin '{plugin_name}': {e}")
            raise

    def get_call_status(self, request_id: str) -> Optional[str]:
        """
        Get the status of an async service call.
        """
        return self.event_bus.get_call_status(request_id)

    def get_call_result(self, request_id: str) -> Optional[Any]:
        """
        Get the result of a completed async service call.
        """
        return self.event_bus.get_call_result(request_id)

    def cancel_call(self, request_id: str) -> bool:
        """
        Cancel an async service call if possible.
        """
        return self.event_bus.cancel_call(request_id)

    def get_available_services(self, plugin_name: str) -> List[str]:
        """
        Get all available services from a specific plugin.
        """
        return self.event_bus.get_plugin_services(plugin_name)

    def register_decorated_services(self) -> None:
        """
        Register all methods decorated with @service in this plugin.
        """
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if hasattr(method, '_is_service'):
                service_name = getattr(method, '_service_name', name)
                self.register_service(service_name, method)
                self.logger.debug(f"Found and registered decorated service: {service_name}")

    def initialize(self) -> None:
        """Initialize the plugin."""
        self.logger.info(f"Initializing {self.name}({self.version})")
        # Register services that were decorated with @service
        self.register_decorated_services()

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        for subscription in self.subscriptions:
            subscription.dispose()
        self.logger.info(f"Shutting down '{self.name}'")

    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
        self.logger.info(f"Enabled '{self.name}'")

    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
        self.logger.info(f"Disabled '{self.name}'")


