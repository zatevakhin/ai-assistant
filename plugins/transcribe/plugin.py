from queue import Queue
from faster_whisper import WhisperModel
from assistant.utils import observe
from assistant.core import Plugin, EventBus
from typing import List
from voice_pulse import SpeechSegment

from assistant.config import (
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_LOCATION,
    WHISPER_USE_COMPUTE_TYPE,
    WHISPER_USE_DEVICE,
)

from .types import TranscribedSegment
from . import events
from plugins.vad.events import VAD_SPEECH_DETECT


class WhisperTranscriber(Plugin):

    def __init__(self, name: str, event_bus: EventBus):
        super().__init__(name, event_bus)
        self.enabled = False

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

        sub = self.event_bus.subscribe(VAD_SPEECH_DETECT, self.on_speech)
        self.subscriptions.append(sub) # cleanup later

        # FIXME: Currently works in single threaded mode.
        self.segments_queue: Queue[SpeechSegment] = Queue()
        sub1 = observe(self.segments_queue, self.on_transcribe)
        self.subscriptions.append(sub1) # cleanup later

        self.whisper = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_USE_DEVICE,
            compute_type=WHISPER_USE_COMPUTE_TYPE,
            download_root=WHISPER_MODELS_LOCATION,
        )

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_speech(self, segment: SpeechSegment):
        self.logger.info(f"> on_speech({type(segment)})")
        self.event_bus.publish(events.TRANSCRIPTION_QUEUE_ADDED, self.segments_queue.qsize() + 1)
        # FIXME: Part of that thing about multi-threading from above.
        # This code part will wait until queue is free before adding new segment.
        self.segments_queue.put(segment)

    def on_transcribe(self, segment: SpeechSegment):
        self.logger.info(f"> on_transcribe({type(segment)})")
        self.event_bus.publish(events.TRANSCRIPTION_SEGMENT_STARTED, None)

        segments, info = self.whisper.transcribe(segment.speech)
        text = "".join(map(lambda s: s.text, filter(lambda s: s.text, segments))).strip()

        transcribed = TranscribedSegment(
            text=text,
            language=info.language,
            probability=info.language_probability,
        )

        self.logger.info(f"> on_transcribe({type(transcribed)}) -> '{transcribed.text}'")
        self.event_bus.publish(events.TRANSCRIPTION_SEGMENT_DONE, transcribed)


