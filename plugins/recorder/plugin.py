import os
from typing import List

import soundfile as sf
from voice_pulse import SpeechSegment

from assistant.config import SPEECH_PIPELINE_SAMPLERATE
from assistant.core import Plugin, service
from plugins.vad.events import VAD_SPEECH_DETECT

from . import events


class SpeechRecorder(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.RECORDER_FILE_SAVED,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.recordings_dir = self.get_config("location", "/tmp/speech-recordings/")

        self.add_dependency("vad")
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
