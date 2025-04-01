from time import sleep
from assistant.components.mumble import MumbleInterface
from assistant.core import EventBus

from assistant.components.mumble import events as mm
from assistant.components.recorder.main import Recorder
from assistant.components.transcriber.main import TranscriberService
from assistant.components.transcriber import events as tt
from assistant.components.system.main import SystemIII
from assistant.components.watchdog.main import Watchdog
from assistant.components.watchdog import events as ww
from assistant.components.shadow.main import Shadow
import logging

from rich.logging import RichHandler
from assistant.core.config_manager import ConfigManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - [yellow]%(threadName)-10s[/] - %(message)s",
    handlers=[RichHandler(show_time=False, markup=True)],
)


def main():
    event_bus = EventBus()
    config = ConfigManager()

    mumble = MumbleInterface(name="mumble", config=config)
    recorder = Recorder(config=config)
    transcriber = TranscriberService(config=config)
    watchdog = Watchdog(name="watchdog", config=config)
    system = SystemIII(config=config)
    shadow = Shadow(config=config)

    event_bus.register(mumble)
    event_bus.register(watchdog)
    event_bus.register(transcriber)
    event_bus.register(system)
    event_bus.register(shadow)

    # mumble.on(mm.MUMBLE_CLIENT_CONNECTED, lambda: print("-> connect"))
    # mumble.on(mm.MUMBLE_CLIENT_DISCONNECTED, lambda: print("-> disconnect"))
    mumble.on(mm.MUMBLE_AUDIO_SPEECH, recorder.on_speech)
    mumble.on(mm.MUMBLE_AUDIO_SPEECH, transcriber.on_speech)
    watchdog.on(ww.WATCHDOG_AUDIO_SPEECH_DETECTED, transcriber.on_speech)
    #transcriber.on(tt.TRANSCRIPTION_SEGMENT_DONE, system.on_transcript)
    transcriber.on(tt.TRANSCRIPTION_SEGMENT_DONE, shadow.on_transcript)


    mumble.initialize()
    recorder.initialize()
    transcriber.initialize()
    system.initialize()
    watchdog.initialize()
    shadow.initialize()

    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            print(end="\r")
            break

    recorder.shutdown()
    mumble.shutdown()
    transcriber.shutdown()
    system.shutdown()
    watchdog.shutdown()
    shadow.shutdown()


if "__main__" == __name__:
    main()
