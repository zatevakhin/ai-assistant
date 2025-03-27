from concurrent.futures import Future
from typing import List

from voice_pulse import SpeechSegment

from assistant.core import Plugin
from plugins.vad.events import VAD_SPEECH_DETECT

from . import events


class SystemIII(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.TRANSCRIPTION_QUEUE_ADDED,
            events.TRANSCRIPTION_SEGMENT_DONE,
            events.TRANSCRIPTION_SEGMENT_STARTED,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.add_dependency("vad")

        self.event_bus.subscribe(VAD_SPEECH_DETECT, lambda x: self.on_speech(*x))

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_speech(self, source: str, segment: SpeechSegment):
        self.logger.info(f"> on_speech({type(segment)})")

        _, future = self.call_service_async("transcriber", "transcribe_speech", segment)
        future.add_done_callback(self.on_transcription_complete)

    def on_transcription_complete(self, future: Future):
        try:
            result = future.result()
            self.logger.info(f"Transcription completed: {result}")
            id, fut = self.call_service_async("shadow", "receive", result.text)

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
