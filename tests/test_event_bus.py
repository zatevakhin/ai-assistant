"""
Tests for the EventBus component.
"""
import pytest
from unittest.mock import MagicMock

from assistant.core.event_bus import EventBus
from assistant.core.service import service


@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for each test."""
    return EventBus()

class TestEventRegistration:
    """Test event registration and subscription."""

    def test_register_event(self, event_bus):
        """Test registering a single event."""
        result = event_bus.register_event("test.event", "test_plugin")
        assert result is True
        assert "test.event" in event_bus.event_registry
        assert event_bus.event_registry["test.event"] == "test_plugin"

    def test_register_duplicate_event_same_plugin(self, event_bus):
        """Test registering the same event twice with the same plugin."""
        event_bus.register_event("test.event", "test_plugin")
        result = event_bus.register_event("test.event", "test_plugin")
        assert result is True  # Should succeed as it's the same plugin

    def test_register_duplicate_event_different_plugin(self, event_bus):
        """Test registering the same event with a different plugin."""
        event_bus.register_event("test.event", "test_plugin1")
        result = event_bus.register_event("test.event", "test_plugin2")
        assert result is False  # Should fail as it's registered by a different plugin

    def test_register_multiple_events(self, event_bus):
        """Test registering multiple events at once."""
        events = ["test.event1", "test.event2", "test.event3"]
        registered = event_bus.register_events(events, "test_plugin")
        assert registered == events
        assert all(e in event_bus.event_registry for e in events)

    def test_get_all_events(self, event_bus):
        """Test getting all registered events."""
        event_bus.register_event("test.event1", "plugin1")
        event_bus.register_event("test.event2", "plugin2")
        events = event_bus.get_all_events()
        assert events == {"test.event1": "plugin1", "test.event2": "plugin2"}


class TestEventPublishSubscribe:
    """Test event publishing and subscription."""

    def test_publish_subscribe(self, event_bus):
        """Test basic publish-subscribe functionality."""
        event_bus.register_event("test.event", "test_plugin")

        mock_handler = MagicMock(wraps=lambda x: x)
        event_bus.subscribe("test.event", mock_handler)

        # Publish an event
        test_data = {"key": "value"}
        event_bus.publish("test.event", test_data)

        # Check the handler was called with the correct data
        mock_handler.assert_called_once_with(test_data)

    def test_multiple_subscribers(self, event_bus):
        """Test that multiple subscribers all receive the event."""
        event_bus.register_event("test.event", "test_plugin")

        # Set up multiple mock subscribers
        mock_handler1 = MagicMock(wraps=lambda x: x)
        mock_handler2 = MagicMock(wraps=lambda x: x)

        event_bus.subscribe("test.event", mock_handler1)
        event_bus.subscribe("test.event", mock_handler2)

        # Publish an event
        test_data = {"key": "value"}
        event_bus.publish("test.event", test_data)

        # Check both handlers were called
        mock_handler1.assert_called_once_with(test_data)
        mock_handler2.assert_called_once_with(test_data)

    def test_unsubscribe(self, event_bus):
        """Test that unsubscribing works correctly."""
        event_bus.register_event("test.event", "test_plugin")

        # Set up a mock subscriber
        mock_handler = MagicMock(wraps=lambda x: x)
        subscription = event_bus.subscribe("test.event", mock_handler)

        # Publish an event
        event_bus.publish("test.event", {"first": True})

        # Unsubscribe
        subscription.dispose()

        # Publish again
        event_bus.publish("test.event", {"second": True})

        # Check the handler was only called once
        assert mock_handler.call_count == 1

    def test_publish_unregistered_event(self, event_bus):
        """Test publishing an event that hasn't been registered."""
        # This should log an error but not raise an exception
        try:
            event_bus.publish("unregistered.event", {"key": "value"})
            assert True  # No exception was raised
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")


class TestServiceRegistration:
    """Test service registration and discovery."""

    def test_register_service(self, event_bus):
        """Test registering a service."""
        mock_method = MagicMock(return_value="result")
        result = event_bus.register_service("test_plugin", "test_service", mock_method)
        assert result is True
        assert "test_plugin" in event_bus.services
        assert "test_service" in event_bus.services["test_plugin"]

    def test_register_duplicate_service(self, event_bus):
        """Test registering the same service twice."""
        mock_method = MagicMock(return_value="result")
        event_bus.register_service("test_plugin", "test_service", mock_method)
        result = event_bus.register_service("test_plugin", "test_service", mock_method)
        assert result is False  # Should fail as it's already registered

    def test_get_plugin_services(self, event_bus):
        """Test getting all services for a plugin."""
        event_bus.register_service("test_plugin", "service1", MagicMock())
        event_bus.register_service("test_plugin", "service2", MagicMock())
        services = event_bus.get_plugin_services("test_plugin")
        assert set(services) == {"service1", "service2"}

    def test_get_all_services(self, event_bus):
        """Test getting all registered services."""
        event_bus.register_service("plugin1", "service1", MagicMock())
        event_bus.register_service("plugin2", "service2", MagicMock())
        all_services = event_bus.get_all_services()
        assert all_services == {"plugin1": ["service1"], "plugin2": ["service2"]}


class TestServiceDecorator:
    """Test the service decorator."""

    def test_service_decorator_detection(self):
        """Test that the service decorator correctly identifies sync and async methods."""

        class TestClass:
            @service
            def sync_method(self):
                return "sync"

            @service
            async def async_method(self):
                return "async"

        instance = TestClass()

        # Check sync method
        assert hasattr(instance.sync_method, '_is_service')
        assert getattr(instance.sync_method, "_is_service") is True

        assert hasattr(instance.sync_method, '_is_async')
        assert getattr(instance.sync_method, "_is_async") is False

        # Check async method
        assert hasattr(instance.async_method, '_is_service')
        assert getattr(instance.async_method, "_is_service") is True
        assert hasattr(instance.async_method, '_is_async')
        assert getattr(instance.async_method, "_is_async") is True

    def test_service_decorator_custom_name(self):
        """Test that the service decorator handles custom names."""

        class TestClass:
            @service("custom_name")
            def method(self):
                return "result"

        instance = TestClass()
        assert hasattr(instance.method, "_service_name") is True
        assert getattr(instance.method, "_service_name") == "custom_name"

