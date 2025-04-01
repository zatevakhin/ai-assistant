from queue import Queue
from typing import List
from assistant.components.mumble.mumble import SpeechSegment
from assistant.components.transcriber.types import Transcript
from assistant.core.component import Component
from assistant.utils.utils import observe
from datetime import datetime
from . import events


class SystemIII(Component):
    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return [
            events.SYSTEM_RECEIVE_TRANSCRIPT,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

        self.transcripts = Queue()
        observe(self.transcripts, lambda args: self.process_transcript(*args))

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_transcript(self, segment: SpeechSegment, transcript: Transcript):
        self.transcripts.put_nowait((segment, transcript))

    def process_transcript(self, segment: SpeechSegment, transcript: Transcript):
        self.logger.info(f"-> {datetime.now() - segment.timestamp}")
        self.logger.info(f"T -> {transcript}")

