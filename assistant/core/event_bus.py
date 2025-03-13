from typing import Dict, List, Any, Callable
from reactivex import Subject
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EventBus:
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}
        self.event_registry: Dict[str, str] = {}  # event_id -> plugin_name

    def register_event(self, event_id: str, plugin_name: str) -> bool:
        """Register an event with the bus. Return True if successful, False if already registered."""
        if event_id in self.event_registry:
            if self.event_registry[event_id] != plugin_name:
                logger.error(f"Event '{event_id}' already registered by plugin {self.event_registry[event_id]}")
                return False
            return True  # Already registered by the same plugin

        self.event_registry[event_id] = plugin_name
        logger.info(f"Registered event '{event_id}' for plugin '{plugin_name}'")
        return True

    def register_events(self, event_ids: List[str], plugin_name: str) -> List[str]:
        """Register multiple events, return list of successfully registered events."""
        registered = []
        for event_id in event_ids:
            if self.register_event(event_id, plugin_name):
                registered.append(event_id)
        return registered

    def get_subject(self, event_id: str) -> Subject:
        """Get or create a subject for an event."""
        if event_id not in self.event_registry:
            raise ValueError(f"Event '{event_id}' is not registered")

        if event_id not in self.subjects:
            self.subjects[event_id] = Subject()
        return self.subjects[event_id]

    def publish(self, event_id: str, data: Any):
        """Publish data to an event subject."""
        try:
            subject = self.get_subject(event_id)
            subject.on_next(data)
            logger.debug(f"Published event '{event_id}'")
        except ValueError as e:
            logger.error(f"Failed to publish event: {e}")

    def subscribe(self, event_id: str, observer: Callable[[Any], None]):
        """Subscribe to an event."""
        try:
            subject = self.get_subject(event_id)
            subscription = subject.subscribe(observer)
            logger.debug(f"Subscribed to event '{event_id}'")
            return subscription
        except ValueError as e:
            logger.error(f"Failed to subscribe to event: {e}")
            return None

    def get_all_events(self) -> Dict[str, str]:
        """Get all registered events and their owning plugins."""
        return self.event_registry.copy()

