from assistant.config import SPEECH_PIPELINE_SAMPLERATE
from assistant.core import Plugin, EventBus, service
from typing import List
from voice_pulse import SpeechSegment
import soundfile as sf
import os

from . import events
from plugins.vad.events import VAD_SPEECH_DETECT

class SpeechRecorder(Plugin):

    def __init__(self, name: str, event_bus: EventBus):
        super().__init__(name, event_bus)
        # NOTE: Debug plugin, don't use in prod.
        self.enabled = True

    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.RECORDER_FILE_SAVED,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.add_dependency("vad")
        self.recordings_dir = "./.recordings/"
        if not os.path.exists(self.recordings_dir):
            os.mkdir(self.recordings_dir)

        sub = self.event_bus.subscribe(VAD_SPEECH_DETECT, lambda x: self.on_speech(*x))
        self.subscriptions.append(sub)

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_speech(self, source: str, segment: SpeechSegment):
        self.logger.info(f"{self.name} > on_speech({type(segment)})")
        path = os.path.join(self.recordings_dir, f"{source}-{segment.timestamp}.flac")
        sf.write(path, segment.speech, samplerate=SPEECH_PIPELINE_SAMPLERATE)
        self.event_bus.publish(events.RECORDER_FILE_SAVED, path)

    @service
    async def store_segment(self, segment: SpeechSegment):

        self.logger.info(f"{self.name} > on_speech({type(segment)})")

