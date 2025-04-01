from assistant.core.component import Component
from typing import Optional, Dict, Any, List
import soundfile as sf
import os
from assistant.config import SPEECH_PIPELINE_SAMPLERATE
from assistant.components.mumble.mumble import SpeechSegment

class Recorder(Component):
    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return []

    def initialize(self) -> None:
        super().initialize()
        self.recordings_dir = self.get_config("location", "/tmp/speech-recordings/")
        if not os.path.exists(self.recordings_dir):
            os.mkdir(self.recordings_dir)

        self.logger.setLevel(self.get_config("log_level", "DEBUG"))
        self.logger.info(f"Plugin '{self.name}' initialized and ready")


    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' disconnection from server.")

    def on_speech(self, segment: SpeechSegment):
        self.logger.info(f"{self.name} ({segment.source}) > on_speech({type(segment)})")
        path = os.path.join(self.recordings_dir, f"{segment.source}-{segment.timestamp}.flac")
        sf.write(path, segment.data, samplerate=SPEECH_PIPELINE_SAMPLERATE)

