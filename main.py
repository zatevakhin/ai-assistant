import time
import logging
from rich.logging import RichHandler

from assistant.core import PluginManager

logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    manager = PluginManager()
    logger.info("Initializing plugin system")
    manager.load_plugins()
    manager.initialize_plugins()

    logger.info("Yolooooooo")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("", end="\r")
        logger.info("Manually interrupted...")

    logger.info("Shutting down plugin system")
    manager.shutdown_plugins()

if "__main__" == __name__:
    main()
