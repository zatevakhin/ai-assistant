import logging
from abc import ABC, abstractmethod
from typing import  List, Any, Set, Callable
from .event_bus import EventBus


class Plugin(ABC):
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self.logger = logging.getLogger(f"plugin.{name}")
        self.logger.setLevel(logging.INFO)
        self.enabled = True
        self.dependencies = set()
        self.subscriptions = []
        self.registered_events = []

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin."""
        pass

    @abstractmethod
    def get_event_definitions(self) -> List[str]:
        """Return a list of event constants defined by this plugin."""
        pass

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

    def initialize(self) -> None:
        """Initialize the plugin."""
        self.logger.info(f"Initializing {self.name}({self.version})")

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
