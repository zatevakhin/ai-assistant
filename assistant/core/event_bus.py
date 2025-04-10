import asyncio
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from reactivex import Subject

from assistant.core.component import Component

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ServiceInfo:
    """Information about a registered service"""

    def __init__(self, method: Callable, is_async: bool):
        self.method = method
        self.is_async = is_async


class EventBus:
    def __init__(self):
        # TODO: Config manager.
        self.subjects: Dict[str, Subject] = {}
        self.event_registry: Dict[str, str] = {}  # event_id -> component_name
        self.services: Dict[
            str, Dict[str, ServiceInfo]
        ] = {}  # component_name -> {service_name -> ServiceInfo}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.pending_calls: Dict[str, Future] = {}  # request_id -> Future

    def register_event(self, event_id: str, component_name: str) -> bool:
        """Register an event with the bus. Return True if successful, False if already registered."""
        if event_id in self.event_registry:
            if self.event_registry[event_id] != component_name:
                logger.error(
                    f"Event '{event_id}' already registered by component {self.event_registry[event_id]}"
                )
                return False
            return True  # Already registered by the same component

        self.event_registry[event_id] = component_name
        logger.info(f"Registered event '{event_id}' for component '{component_name}'")
        return True

    def register(self, component: Component):
        self.register_events(component.events, component.name)

        for service, method in component.get_services():
            self.register_service(component.name, service, method)

    def register_events(self, event_ids: List[str], component_name: str) -> List[str]:
        """Register multiple events, return list of successfully registered events."""
        registered = []
        for event_id in event_ids:
            if self.register_event(event_id, component_name):
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
        """Get all registered events and their owning components."""
        return self.event_registry.copy()

    def register_service(
        self, component_name: str, service_name: str, method: Callable
    ) -> bool:
        """
        Register a service method with the bus.
        """
        if component_name not in self.services:
            self.services[component_name] = {}

        if service_name in self.services[component_name]:
            logger.warning(
                f"Service '{service_name}' already registered for component '{component_name}'"
            )
            return False

        is_async = hasattr(method, "_is_async") and getattr(method, "_is_async")
        self.services[component_name][service_name] = ServiceInfo(method, is_async)
        logger.info(
            f"Registered service '{service_name}' ({'async' if is_async else 'sync'}) for component '{component_name}'"
        )
        return True

    def call_service(self, component_name: str, service_name: str, *args, **kwargs) -> Any:
        """
        Synchronously call a service method on a component.
        This method should ONLY be used for synchronous service methods.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Service call {request_id}: {component_name}.{service_name}")

        service_info = self._get_service_info(component_name, service_name)

        if service_info.is_async:
            raise ValueError(
                f"Service '{service_name}' on component '{component_name}' is async. Use call_service_async instead."
            )

        try:
            # Call the synchronous service method directly
            return service_info.method(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error calling service '{service_name}' on component '{component_name}' (request {request_id}): {e}"
            )
            raise

    def call_service_async(
        self, component_name: str, service_name: str, *args, **kwargs
    ) -> Tuple[str, Future]:
        """
        Asynchronously call a service method on a component.
        This method should ONLY be used for asynchronous service methods.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Async service call {request_id}: {component_name}.{service_name}")

        service_info = self._get_service_info(component_name, service_name)

        if not service_info.is_async:
            raise ValueError(
                f"Service '{service_name}' on component '{component_name}' is not async. Use call_service instead."
            )

        def async_wrapper():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(service_info.method(*args, **kwargs))
            finally:
                loop.close()

        future = self.thread_pool.submit(async_wrapper)

        self.pending_calls[request_id] = future

        future.add_done_callback(lambda _: self._cleanup_call(request_id))

        return request_id, future

    def _cleanup_call(self, request_id: str) -> None:
        """Remove a completed call from pending calls."""
        if request_id in self.pending_calls:
            del self.pending_calls[request_id]

    def _get_service_info(self, component_name: str, service_name: str) -> ServiceInfo:
        """Get service info, raising appropriate errors if not found."""
        if component_name not in self.services:
            raise ValueError(f"Component '{component_name}' has no registered services")

        if service_name not in self.services[component_name]:
            raise ValueError(
                f"Service '{service_name}' not found in component '{component_name}'"
            )

        return self.services[component_name][service_name]

    def get_call_status(self, request_id: str) -> Optional[str]:
        """
        Get the status of an async service call.
        """
        if request_id not in self.pending_calls:
            return None

        future = self.pending_calls[request_id]
        if future.done():
            if future.exception():
                return "error"
            return "completed"
        return "running"

    def get_call_result(self, request_id: str) -> Optional[Any]:
        """
        Get the result of a completed async service call.
        """
        if request_id not in self.pending_calls:
            return None

        future = self.pending_calls[request_id]
        if not future.done():
            return None

        # This will raise the exception if there was one
        return future.result()

    def cancel_call(self, request_id: str) -> bool:
        """
        Cancel an async service call if possible.
        """
        if request_id not in self.pending_calls:
            return False

        future = self.pending_calls[request_id]
        result = future.cancel()
        if result:
            del self.pending_calls[request_id]
        return result

    def get_service(self, component_name: str, service_name: str) -> Optional[Callable]:
        """
        Get a service method reference.
        """
        if component_name not in self.services:
            return None

        service_info = self.services[component_name].get(service_name)
        if service_info:
            return service_info.method
        return None

    def get_all_services(self) -> Dict[str, List[str]]:
        """
        Get all registered services and their owning components.
        """
        return {
            component: list(services.keys()) for component, services in self.services.items()
        }

    def get_component_services(self, component_name: str) -> List[str]:
        """
        Get all services registered by a specific component.
        """
        if component_name not in self.services:
            return []

        return list(self.services[component_name].keys())
