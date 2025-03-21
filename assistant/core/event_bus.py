from typing import Dict, List, Any, Callable, Optional, Tuple
from reactivex import Subject
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ServiceInfo:
    """Information about a registered service"""
    def __init__(self, method: Callable, is_async: bool):
        self.method = method
        self.is_async = is_async

class EventBus:
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}
        self.event_registry: Dict[str, str] = {}  # event_id -> plugin_name
        self.services: Dict[str, Dict[str, ServiceInfo]] = {}  # plugin_name -> {service_name -> ServiceInfo}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.pending_calls: Dict[str, Future] = {}  # request_id -> Future

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

    def register_service(self, plugin_name: str, service_name: str, method: Callable) -> bool:
        """
        Register a service method with the bus.
        """
        if plugin_name not in self.services:
            self.services[plugin_name] = {}

        if service_name in self.services[plugin_name]:
            logger.warning(f"Service '{service_name}' already registered for plugin '{plugin_name}'")
            return False

        is_async = getattr(method, "_is_async")
        self.services[plugin_name][service_name] = ServiceInfo(method, is_async)
        logger.info(f"Registered service '{service_name}' ({'async' if is_async else 'sync'}) for plugin '{plugin_name}'")
        return True

    def call_service(self, plugin_name: str, service_name: str, *args, **kwargs) -> Any:
        """
        Synchronously call a service method on a plugin.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Service call {request_id}: {plugin_name}.{service_name}")

        service_info = self._get_service_info(plugin_name, service_name)

        try:
            if service_info.is_async:
                # For async methods, we need to run them in an event loop
                return self._run_coroutine(service_info.method, *args, **kwargs)
            else:
                # For sync methods, call directly
                return service_info.method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error calling service '{service_name}' on plugin '{plugin_name}' (request {request_id}): {e}")
            raise

    def call_service_async(self, plugin_name: str, service_name: str, *args, **kwargs) -> Tuple[str, Future]:
        """
        Asynchronously call a service method on a plugin.
        """
        request_id = str(uuid.uuid4())
        logger.debug(f"Async service call {request_id}: {plugin_name}.{service_name}")

        service_info = self._get_service_info(plugin_name, service_name)

        if service_info.is_async:
            # For async methods, submit the coroutine runner to the thread pool
            future = self.thread_pool.submit(
                self._run_coroutine, service_info.method, *args, **kwargs
            )
        else:
            # For sync methods, just submit the call to the thread pool
            future = self.thread_pool.submit(service_info.method, *args, **kwargs)

        self.pending_calls[request_id] = future

        # Set cleanup callback
        future.add_done_callback(lambda _: self._cleanup_call(request_id))

        return request_id, future

    def _run_coroutine(self, coro_func, *args, **kwargs):
        """Run a coroutine function and return its result."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_func(*args, **kwargs))
        finally:
            loop.close()

    def _cleanup_call(self, request_id: str) -> None:
        """Remove a completed call from pending calls."""
        if request_id in self.pending_calls:
            del self.pending_calls[request_id]

    def _get_service_info(self, plugin_name: str, service_name: str) -> ServiceInfo:
        """Get service info, raising appropriate errors if not found."""
        if plugin_name not in self.services:
            raise ValueError(f"Plugin '{plugin_name}' has no registered services")

        if service_name not in self.services[plugin_name]:
            raise ValueError(f"Service '{service_name}' not found in plugin '{plugin_name}'")

        return self.services[plugin_name][service_name]

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

    def get_service(self, plugin_name: str, service_name: str) -> Optional[Callable]:
        """
        Get a service method reference.
        """
        if plugin_name not in self.services:
            return None

        service_info = self.services[plugin_name].get(service_name)
        if service_info:
            return service_info.method
        return None

    def get_all_services(self) -> Dict[str, List[str]]:
        """
        Get all registered services and their owning plugins.
        """
        return {plugin: list(services.keys()) for plugin, services in self.services.items()}

    def get_plugin_services(self, plugin_name: str) -> List[str]:
        """
        Get all services registered by a specific plugin.
        """
        if plugin_name not in self.services:
            return []

        return list(self.services[plugin_name].keys())

