from faster_whisper import WhisperModel
import zenoh
from assistant.config import (
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_LOCATION,
    WHISPER_USE_COMPUTE_TYPE,
    WHISPER_USE_DEVICE,
    TOPIC_VAD_SPEECH_NEW,
    TOPIC_TRANSCRIPTION_DONE,
    ZENOH_CONFIG,
)
from queue import Queue
import logging
import numpy as np
from numpy.typing import NDArray
from .event_bus import EventBus
from voice_pulse import SpeechSegment
from reactivex.scheduler import NewThreadScheduler
from .util import queue_as_observable
from functools import partial

logger = logging.getLogger(__name__)

class SpeechTranscriberProcess:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.zenoh_session = zenoh.open(ZENOH_CONFIG)

        self.transcriber_scheduler = NewThreadScheduler()
        self.speech_subscription = self.event_bus.subscribe(TOPIC_VAD_SPEECH_NEW, self.on_speech)

        self.speech_queue: Queue[NDArray[np.float32]] = Queue(maxsize=1)
        self.observable_speech = queue_as_observable(self.speech_queue)

        self.on_transcription = partial(self.event_bus.publish, TOPIC_TRANSCRIPTION_DONE)
        self.observable_speech.subscribe(self.__speech_transcribe)

        self.pub_transcription_done = self.zenoh_session.declare_publisher(TOPIC_TRANSCRIPTION_DONE)

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
        logger.debug(f"{type(speech)}, {speech.shape}, {speech.dtype}, writeable: {speech.flags.writeable}")
        segments, info = self.whisper.transcribe(speech)
        text = "".join(map(lambda s: s.text, filter(lambda s: s.text, segments))).strip()

        transcription = {
            "text": text,
            "language": info.language,
            "probability": info.language_probability
        }

        self.on_transcription(transcription)
        self.pub_transcription_done.put(transcription)
