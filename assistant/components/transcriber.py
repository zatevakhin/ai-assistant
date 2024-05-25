from faster_whisper import WhisperModel
import zenoh
from assistant.config import (
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_LOCATION,
    WHISPER_USE_COMPUTE_TYPE,
    WHISPER_USE_DEVICE,
    TOPIC_VAD_SPEECH_NEW,
    TOPIC_TRANSCRIPTION_DONE,
)
from queue import Queue, Empty
import logging
import numpy as np
from zenoh import Sample
import threading

logger = logging.getLogger(__name__)

class SpeechTranscriberProcess:
    def __init__(self):
        self.zenoh_session = zenoh.open({
            "connect": {
                "endpoints": ["tcp/localhost:7447"],
            },
        })
        self.sub_mumble_sound = self.zenoh_session.declare_subscriber(TOPIC_VAD_SPEECH_NEW, self.on_new_speech)
        self.pub_transcription_done = self.zenoh_session.declare_publisher(TOPIC_TRANSCRIPTION_DONE)

        self.whisper = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_USE_DEVICE,
            compute_type=WHISPER_USE_COMPUTE_TYPE,
            download_root=WHISPER_MODELS_LOCATION,
        )

        self.speech_queue: Queue[np.ndarray[np.float32]] = Queue(maxsize=1)
        self.running = False
        self.thread = threading.Thread(target=self._speech_transcribe)

        logger.info("Whisper Transcriber ... IDLE")

    def run(self):
        self.running = True
        self.thread.start()
        logger.info("Whisper Transcriber ... OK")

    def stop(self):
        logger.info("Whisper Transcriber ... STOPPING")
        self.running = False
        self.thread.join()
        logger.info("Whisper Transcriber ... DEAD")

    def on_new_speech(self, sample: Sample):
        logger.info(f"> on_new_speech({type(sample)})")

        speech = np.frombuffer(sample.payload, dtype=np.float64)
        speech = np.array(speech) # NOTE: Convert to Writable numpy array.
        self.speech_queue.put(speech)

    def _speech_transcribe(self):
        while self.running:
            try:
                speech = self.speech_queue.get(timeout=1)
            except Empty:
                continue

            logger.debug(f"{type(speech)}, {speech.shape}, {speech.dtype}, writeable: {speech.flags.writeable}")
            segments, info = self.whisper.transcribe(speech)
            text = "".join(map(lambda s: s.text, filter(lambda s: s.text, segments))).strip()

            self.pub_transcription_done.put({
                "text": text,
                "language": info.language,
                "probability": info.language_probability
            })

            self.speech_queue.task_done()
