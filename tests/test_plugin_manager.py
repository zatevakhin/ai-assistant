"""
Tests for the Plugin and PluginManager components.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, call
from assistant.core.event_bus import EventBus
from assistant.core.plugin import Plugin
from assistant.core.plugin_manager import PluginManager
from assistant.core.service import service


# --- Plugin class implementations for testing ---


class ExamplePlugin(Plugin):
    """A basic test plugin implementation."""

    @property
    def version(self):
        return "1.0.0"

    def get_event_definitions(self):
        return ["test.event1", "test.event2"]

    @service
    def test_service(self, arg):
        return f"Service result: {arg}"

    @service
    async def async_service(self, arg):
        await asyncio.sleep(0.1)
        return f"Async service result: {arg}"


class DependentPlugin(Plugin):
    """A plugin with dependencies on TestPlugin."""

    @property
    def version(self):
        return "0.5.0"

    def get_event_definitions(self):
        return ["dependent.event"]

    def initialize(self):
        super().initialize()
        self.add_dependency("test_plugin")


class CircularPlugin1(Plugin):
    """A plugin with circular dependency (part 1)."""

    @property
    def version(self):
        return "0.1.0"

    def get_event_definitions(self):
        return ["circular1.event"]

    def initialize(self):
        super().initialize()
        self.add_dependency("circular2")


class CircularPlugin2(Plugin):
    """A plugin with circular dependency (part 2)."""

    @property
    def version(self):
        return "0.1.0"

    def get_event_definitions(self):
        return ["circular2.event"]

    def initialize(self):
        super().initialize()
        self.add_dependency("circular1")


# --- Fixtures ---


@pytest.fixture
def event_bus():
    """Create a fresh EventBus instance for each test."""
    return EventBus()


@pytest.fixture
def test_plugin(event_bus):
    """Create a TestPlugin instance."""
    return ExamplePlugin("test_plugin", event_bus)


@pytest.fixture
def dependent_plugin(event_bus):
    """Create a DependentPlugin instance."""
    return DependentPlugin("dependent_plugin", event_bus)


@pytest.fixture
def circular_plugin1(event_bus):
    """Create a CircularPlugin1 instance."""
    return CircularPlugin1("circular1", event_bus)


@pytest.fixture
def circular_plugin2(event_bus):
    """Create a CircularPlugin2 instance."""
    return CircularPlugin2("circular2", event_bus)


@pytest.fixture
def plugin_manager():
    """Create a PluginManager instance."""
    return PluginManager()


@pytest.fixture
def populated_plugin_manager(plugin_manager, test_plugin, dependent_plugin):
    """Create a PluginManager with some plugins pre-loaded."""
    plugin_manager.plugins[test_plugin.name] = test_plugin
    plugin_manager.plugins[dependent_plugin.name] = dependent_plugin
    return plugin_manager


# --- Test Plugin class ---


class TestPluginClass:
    """Test the basic Plugin class functionality."""

    def test_plugin_initialization(self, test_plugin):
        """Test plugin initialization."""
        assert test_plugin.name == "test_plugin"
        assert test_plugin.version == "1.0.0"
        assert test_plugin.enabled is True
        assert len(test_plugin.dependencies) == 0

    def test_plugin_register_events(self, test_plugin):
        """Test event registration."""
        # Mock the event_bus.register_events method
        test_plugin.event_bus.register_events = MagicMock(
            return_value=["test.event1", "test.event2"]
        )

        test_plugin.register_events()
        test_plugin.event_bus.register_events.assert_called_once_with(
            ["test.event1", "test.event2"], "test_plugin"
        )
        assert test_plugin.registered_events == ["test.event1", "test.event2"]

    def test_plugin_add_dependency(self, test_plugin):
        """Test adding dependencies."""
        test_plugin.add_dependency("other_plugin")
        assert "other_plugin" in test_plugin.dependencies
        assert "other_plugin" in test_plugin.required_plugins

    def test_plugin_publish_event(self, test_plugin):
        """Test event publishing."""
        # Setup
        test_plugin.registered_events = ["test.event1"]
        test_plugin.event_bus.publish = MagicMock()

        # Test successful publish
        test_plugin.publish_event("test.event1", {"key": "value"})
        test_plugin.event_bus.publish.assert_called_once_with(
            "test.event1", {"key": "value"}
        )

        # Test publishing unregistered event
        test_plugin.event_bus.publish.reset_mock()
        test_plugin.publish_event("unregistered.event", {"key": "value"})
        test_plugin.event_bus.publish.assert_not_called()

        # Test publishing when disabled
        test_plugin.event_bus.publish.reset_mock()
        test_plugin.enabled = False
        test_plugin.publish_event("test.event1", {"key": "value"})
        test_plugin.event_bus.publish.assert_not_called()

    def test_plugin_subscribe_to_event(self, test_plugin):
        """Test event subscription."""
        handler = MagicMock()
        subscription = MagicMock()
        test_plugin.event_bus.subscribe = MagicMock(return_value=subscription)

        test_plugin.subscribe_to_event("some.event", handler)
        test_plugin.event_bus.subscribe.assert_called_once_with("some.event", handler)
        assert subscription in test_plugin.subscriptions

    def test_plugin_enable_disable(self, test_plugin):
        """Test enabling and disabling a plugin."""
        test_plugin.disable()
        assert test_plugin.enabled is False

        test_plugin.enable()
        assert test_plugin.enabled is True

    def test_plugin_shutdown(self, test_plugin):
        """Test plugin shutdown."""
        # Setup subscriptions
        subscription1 = MagicMock()
        subscription2 = MagicMock()
        test_plugin.subscriptions = [subscription1, subscription2]

        # Shutdown
        test_plugin.shutdown()

        # Verify all subscriptions were disposed
        subscription1.dispose.assert_called_once()
        subscription2.dispose.assert_called_once()


# --- Test Plugin RPC functionality ---


class TestPluginRPC:
    """Test the Plugin RPC functionality."""

    def test_register_service(self, test_plugin):
        """Test registering a service."""
        # Mock the event_bus.register_service method
        test_plugin.event_bus.register_service = MagicMock(return_value=True)

        method = MagicMock()
        result = test_plugin.register_service("test_service", method)
        assert result is True
        test_plugin.event_bus.register_service.assert_called_once_with(
            "test_plugin", "test_service", method
        )
        assert "test_service" in test_plugin.registered_services

    def test_register_service_when_disabled(self, test_plugin):
        """Test registering a service when the plugin is disabled."""
        test_plugin.enabled = False
        result = test_plugin.register_service("test_service", MagicMock())
        assert result is False

    def test_call_service(self, test_plugin):
        """Test calling a service on another plugin."""
        # Mock the event_bus.call_service method
        test_plugin.event_bus.call_service = MagicMock(return_value="service result")

        result = test_plugin.call_service(
            "other_plugin", "some_service", "arg1", kwarg1="value"
        )
        assert result == "service result"
        test_plugin.event_bus.call_service.assert_called_once_with(
            "other_plugin", "some_service", "arg1", kwarg1="value"
        )

    def test_call_service_when_disabled(self, test_plugin):
        """Test calling a service when the plugin is disabled."""
        test_plugin.enabled = False
        with pytest.raises(RuntimeError, match="Plugin .* is disabled"):
            test_plugin.call_service("other_plugin", "some_service")

    def test_call_service_async(self, test_plugin):
        """Test calling a service asynchronously."""
        # Mock the event_bus.call_service_async method
        mock_future = MagicMock()
        test_plugin.event_bus.call_service_async = MagicMock(
            return_value=("request-id", mock_future)
        )

        request_id, future = test_plugin.call_service_async(
            "other_plugin", "some_service", "arg1"
        )
        assert request_id == "request-id"
        assert future == mock_future
        test_plugin.event_bus.call_service_async.assert_called_once_with(
            "other_plugin", "some_service", "arg1"
        )

    def test_call_service_async_when_disabled(self, test_plugin):
        """Test calling a service asynchronously when the plugin is disabled."""
        test_plugin.enabled = False
        with pytest.raises(RuntimeError, match="Plugin .* is disabled"):
            test_plugin.call_service_async("other_plugin", "some_service")

    def test_register_decorated_services(self, test_plugin):
        """Test automatic registration of decorated services."""
        # Mock the register_service method
        test_plugin.register_service = MagicMock()

        # Call the method that registers decorated services
        test_plugin.register_decorated_services()

        # Check that both sync and async services were registered
        assert test_plugin.register_service.call_count == 2
        calls = [
            call("test_service", test_plugin.test_service),
            call("async_service", test_plugin.async_service),
        ]
        test_plugin.register_service.assert_has_calls(calls, any_order=True)

    def test_get_available_services(self, test_plugin):
        """Test getting available services from another plugin."""
        test_plugin.event_bus.get_plugin_services = MagicMock(
            return_value=["service1", "service2"]
        )

        services = test_plugin.get_available_services("other_plugin")
        assert services == ["service1", "service2"]
        test_plugin.event_bus.get_plugin_services.assert_called_once_with(
            "other_plugin"
        )

    def test_get_call_status(self, test_plugin):
        """Test getting the status of an async call."""
        test_plugin.event_bus.get_call_status = MagicMock(return_value="running")

        status = test_plugin.get_call_status("request-id")
        assert status == "running"
        test_plugin.event_bus.get_call_status.assert_called_once_with("request-id")

    def test_get_call_result(self, test_plugin):
        """Test getting the result of an async call."""
        test_plugin.event_bus.get_call_result = MagicMock(return_value="result")

        result = test_plugin.get_call_result("request-id")
        assert result == "result"
        test_plugin.event_bus.get_call_result.assert_called_once_with("request-id")

    def test_cancel_call(self, test_plugin):
        """Test cancelling an async call."""
        test_plugin.event_bus.cancel_call = MagicMock(return_value=True)

        result = test_plugin.cancel_call("request-id")
        assert result is True
        test_plugin.event_bus.cancel_call.assert_called_once_with("request-id")


# --- Test PluginManager class ---


class TestPluginManager:
    """Test the PluginManager class."""

    def test_plugin_manager_initialization(self, plugin_manager):
        """Test plugin manager initialization."""
        assert isinstance(plugin_manager.event_bus, EventBus)
        assert plugin_manager.plugins == {}
        assert plugin_manager.plugin_dirs == ["plugins"]

    def test_get_plugin(self, populated_plugin_manager):
        """Test getting a plugin by name."""
        plugin = populated_plugin_manager.get_plugin("test_plugin")
        assert plugin is not None
        assert plugin.name == "test_plugin"

        # Nonexistent plugin
        plugin = populated_plugin_manager.get_plugin("nonexistent")
        assert plugin is None

    def test_register_plugin_events(self, populated_plugin_manager):
        """Test registering events for all plugins."""
        # Mock the register_events method of each plugin
        for plugin in populated_plugin_manager.plugins.values():
            plugin.register_events = MagicMock()

        populated_plugin_manager.register_plugin_events()

        # Verify each plugin's register_events was called
        for plugin in populated_plugin_manager.plugins.values():
            plugin.register_events.assert_called_once()

    def test_resolve_dependencies_simple(self, populated_plugin_manager):
        """Test resolving simple dependencies."""
        # The dependent_plugin depends on test_plugin, so test_plugin should come first
        order = populated_plugin_manager.resolve_dependencies()
        assert order.index("test_plugin") < order.index("dependent_plugin")

    def test_resolve_dependencies_missing(self, populated_plugin_manager):
        """Test resolving dependencies with a missing dependency."""
        # Add a plugin with a missing dependency
        missing_dep_plugin = ExamplePlugin(
            "missing_dep_plugin", populated_plugin_manager.event_bus
        )
        missing_dep_plugin.add_dependency("nonexistent")
        populated_plugin_manager.plugins["missing_dep_plugin"] = missing_dep_plugin

        # Mock the disable method
        missing_dep_plugin.disable = MagicMock()

        populated_plugin_manager.resolve_dependencies()

        # Verify the plugin was disabled due to missing dependency
        missing_dep_plugin.disable.assert_called_once()

    def test_resolve_dependencies_circular(
        self, plugin_manager, circular_plugin1, circular_plugin2
    ):
        """Test resolving circular dependencies."""
        # Add plugins with circular dependencies
        plugin_manager.plugins["circular1"] = circular_plugin1
        plugin_manager.plugins["circular2"] = circular_plugin2

        # Mock the disable method
        circular_plugin1.disable = MagicMock()
        circular_plugin2.disable = MagicMock()

        order = plugin_manager.resolve_dependencies()

        # Verify at least one plugin was disabled due to circular dependency
        assert circular_plugin1.disable.called or circular_plugin2.disable.called

    def test_initialize_plugins(self, populated_plugin_manager):
        """Test initializing all plugins."""
        # Mock methods
        populated_plugin_manager.register_plugin_events = MagicMock()
        populated_plugin_manager.resolve_dependencies = MagicMock(
            return_value=["test_plugin", "dependent_plugin"]
        )

        for plugin in populated_plugin_manager.plugins.values():
            plugin.initialize = MagicMock()

        # Initialize plugins
        populated_plugin_manager.initialize_plugins()

        # Verify methods were called in the right order
        populated_plugin_manager.register_plugin_events.assert_called_once()
        populated_plugin_manager.resolve_dependencies.assert_called_once()

        # Verify each plugin's initialize was called
        for plugin in populated_plugin_manager.plugins.values():
            plugin.initialize.assert_called_once()

    def test_shutdown_plugins(self, populated_plugin_manager):
        """Test shutting down all plugins."""
        # Mock methods
        populated_plugin_manager.resolve_dependencies = MagicMock(
            return_value=["test_plugin", "dependent_plugin"]
        )

        for plugin in populated_plugin_manager.plugins.values():
            plugin.shutdown = MagicMock()

        # Shutdown plugins
        populated_plugin_manager.shutdown_plugins()

        # Verify each plugin's shutdown was called
        for plugin in populated_plugin_manager.plugins.values():
            plugin.shutdown.assert_called_once()

    @pytest.mark.parametrize("init_fails", [True, False])
    def test_initialize_plugin_error(self, populated_plugin_manager, init_fails):
        """Test handling errors during plugin initialization."""
        # Setup
        plugin = populated_plugin_manager.plugins["test_plugin"]

        if init_fails:
            # Make initialize raise an exception
            plugin.initialize = MagicMock(side_effect=Exception("Initialization error"))
            plugin.disable = MagicMock()
        else:
            plugin.initialize = MagicMock()

        # Mock other methods to isolate the test
        populated_plugin_manager.register_plugin_events = MagicMock()
        populated_plugin_manager.resolve_dependencies = MagicMock(
            return_value=["test_plugin"]
        )

        # Initialize plugins
        populated_plugin_manager.initialize_plugins()

        # Verify behavior
        plugin.initialize.assert_called_once()
        if init_fails:
            plugin.disable.assert_called_once()

    def test_plugin_discovery(self, plugin_manager):
        """Test discovering plugins in directory."""
        # This test would normally need actual plugin directories to test
        # In a real test, you might use a temporary directory or mock filesystem
        # For now, we'll just mock the discovery method
        mock_plugins = [("plugin1", ExamplePlugin), ("plugin2", DependentPlugin)]
        plugin_manager.discover_plugins = MagicMock(return_value=mock_plugins)

        with (
            patch.object(ExamplePlugin, "__init__", return_value=None),
            patch.object(DependentPlugin, "__init__", return_value=None),
        ):
            plugin_manager.load_plugins()

        # Verify plugins were loaded
        assert "plugin1" in plugin_manager.plugins
        assert "plugin2" in plugin_manager.plugins

        plugin_manager.discover_plugins.assert_called_once()
