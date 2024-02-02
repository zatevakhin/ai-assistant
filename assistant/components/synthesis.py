import logging
import threading
from queue import Queue, Empty
import resampy
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE

from assistant.config import (
    PIPER_TTS_MODEL,
    PIPER_MODELS_LOCATION,
    TOPIC_LLM_ON_SENTENCE,
    TOPIC_MUMBLE_PLAY_AUDIO,
    TOPIC_SPEECH_SYNTHESIS_INTERRUPT,
)

from voice_forge import PiperTts
from zenoh import Sample
import zenoh
import numpy as np
from .audio import audio_length

logger = logging.getLogger(__name__)


class SpeechSynthesisProcess:
    def __init__(self) -> None:
        self.zenoh_session = zenoh.open(zenoh.Config())
        self.sub_on_sentence = self.zenoh_session.declare_subscriber(TOPIC_LLM_ON_SENTENCE, self.on_sentence)
        self.sub_on_interrupt = self.zenoh_session.declare_subscriber(TOPIC_SPEECH_SYNTHESIS_INTERRUPT, self.on_interruption)
        self.pub_play_audio = self.zenoh_session.declare_publisher(TOPIC_MUMBLE_PLAY_AUDIO)

        self.tts = self.create_tts(PIPER_TTS_MODEL)
        self.sentences_queue: Queue[str] = Queue()

        self.thread = threading.Thread(target=self._synthesise_speech)
        self.is_synthesise = threading.Event()
        self.running = False
        logger.info("Synthesis ... IDLE")

    def create_tts(self, model: str):
        return PiperTts(model, PIPER_MODELS_LOCATION)

    def on_sentence(self, sample: Sample):
        logger.info(f"> on_sentence({sample.payload})")
        self.sentences_queue.put(sample.payload.decode())

    def on_interruption(self, sample: Sample):
        logger.warning(f"> on_interruption({sample.payload})")
        logger.info(f"Sentences to synth: {self.sentences_queue.qsize()}")

    def _synthesise_speech(self):
        while self.running:
            try:
                sentence = self.sentences_queue.get(timeout=1)
                self.is_synthesise.set()
            except Empty:
                continue

            speech, samplerate = self.tts.synthesize_stream(sentence)
            speech_resampled = self._resample_speec_for_mumble(speech, samplerate)

            # length = audio_length(speech_resampled, PYMUMBLE_SAMPLERATE)
            # logger.info(f"Sentence: '{sentence}' is {length}ms long")
            self.pub_play_audio.put(speech_resampled.tobytes())

            self.sentences_queue.task_done()

            # Always clear state, after synth finished or interrupted.
            self.is_synthesise.clear()

    def _resample_speec_for_mumble(self, speech: np.ndarray[np.int16], samplerate: int) -> np.ndarray[np.int16]:
        return resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)

    def run(self):
        self.running = True
        self.thread.start()
        logger.info("Synthesis ... OK")

    def stop(self):
        logger.info("Synthesis ... STOPPING")
        self.running = False
        self.thread.join()
        logger.info("Synthesis ... DEAD")
