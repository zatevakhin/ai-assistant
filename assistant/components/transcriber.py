from faster_whisper import WhisperModel
from assistant.config import (
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_LOCATION,
    WHISPER_USE_COMPUTE_TYPE,
    WHISPER_USE_DEVICE,
)
from queue import Queue
import logging
import numpy as np
from numpy.typing import NDArray
from .event_bus import EventBus, EventType
from voice_pulse import SpeechSegment
from reactivex.scheduler import NewThreadScheduler
from .util import queue_as_observable, controlled_area
from functools import partial
from pydantic import BaseModel, Field
from uuid import uuid4, UUID

logger = logging.getLogger(__name__)

class TranscribedSegment(BaseModel):
    text: str
    language: str
    probability: float
    uuid: UUID = Field(default_factory=uuid4)


class SpeechTranscriberProcess:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.transcriber_scheduler = NewThreadScheduler()
        self.speech_subscription = self.event_bus.subscribe(EventType.VAD_NEW_SPEECH, self.on_speech)

        self.speech_queue: Queue[NDArray[np.float32]] = Queue(maxsize=1)
        self.observable_speech = queue_as_observable(self.speech_queue)

        self.on_transcription = partial(self.event_bus.publish, EventType.TRANSCRIPTION_DONE)
        self.observable_speech.subscribe(self.__speech_transcribe)

        self.whisper = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_USE_DEVICE,
            compute_type=WHISPER_USE_COMPUTE_TYPE,
            download_root=WHISPER_MODELS_LOCATION,
        )

        logger.info("Whisper Transcriber ... IDLE")

    def run(self):
        logger.info("Whisper Transcriber ... OK")

    def stop(self):
        logger.info("Whisper Transcriber ... STOPPING")
        if hasattr(self, 'speech_subscription'):
            self.speech_subscription.dispose()

        logger.info("Whisper Transcriber ... DEAD")

    def on_speech(self, segment: SpeechSegment):
        logger.debug(f"> on_speech({type(segment)})")
        self.speech_queue.put(np.array(segment.speech))

    def __speech_transcribe(self, speech: NDArray[np.float32]):
        with controlled_area(partial(self.event_bus.publish, EventType.TRANSCRIPTION_STATUS), "running", "done", True, __name__):
            logger.debug(f"{type(speech)}, {speech.shape}, {speech.dtype}, writeable: {speech.flags.writeable}")
            segments, info = self.whisper.transcribe(speech)
            text = "".join(map(lambda s: s.text, filter(lambda s: s.text, segments))).strip()

            segment = TranscribedSegment(
                text=text,
                language=info.language,
                probability=info.language_probability,
            )

            self.on_transcription(segment)
