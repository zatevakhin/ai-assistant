import time
import logging
import click
from rich.logging import RichHandler

from assistant.core import PluginManager, ConfigManager

logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@click.group()
@click.option('--config', '-c', default='config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """AI assistant command line interface."""
    # Store config path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config

@cli.command()
@click.pass_context
def run(ctx):
    """Run the AI assistant."""

    config_path = ctx.obj['config_path']
    manager = PluginManager(config_path=config_path)
    logger.info(f"Initializing plugin system with config: {config_path}")
    manager.load_plugins()
    manager.initialize_plugins()

    logger.info("Initialization complete!")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("", end="\r")
        logger.info("Manually interrupted...")

    logger.info("Shutting down plugin system")
    manager.shutdown_plugins()

@cli.command()
@click.argument('plugin_name', required=False)
@click.pass_context
def list_plugins(ctx, plugin_name):
    """List available plugins or show details of a specific plugin."""
    config_path = ctx.obj['config_path']
    config_manager = ConfigManager(config_path)

    # Create plugin manager but don't initialize plugins
    manager = PluginManager(config_path=config_path)
    manager.load_plugins()

    if plugin_name:
        plugin = manager.get_plugin(plugin_name)
        if not plugin:
            logger.info(f"Plugin '{plugin_name}' not found")
            return

        logger.info(f"Plugin: {plugin_name}")
        logger.info(f" - Version: {plugin.version}")
        logger.info(f" - Enabled: {config_manager.is_plugin_enabled(plugin_name)}")
        logger.info(f" - Dependencies: {', '.join(plugin.required_plugins) or 'None'}")
    else:
        logger.info("Available plugins:")
        for name, plugin in manager.plugins.items():
            enabled = "[[green]✓[/green]]" if config_manager.is_plugin_enabled(name) else "[[red]✗[/red]]"
            logger.info(f" - {enabled} {name} ({plugin.version})", extra={"markup": True})

if "__main__" == __name__:
    cli(obj={})

