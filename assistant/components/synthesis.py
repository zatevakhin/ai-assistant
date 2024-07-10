import logging
import threading
from queue import Queue
import resampy
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from typing import Any

from assistant.config import (
    PIPER_TTS_MODEL,
    PIPER_MODELS_LOCATION
)

from .audio import enrich_with_silence
from voice_forge import PiperTts
from .util import queue_as_observable, controlled_area
import numpy as np
from .event_bus import EventBus, EventType
from functools import partial

logger = logging.getLogger(__name__)


class SpeechSynthesisProcess:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.sentence_subscription = self.event_bus.subscribe(EventType.LLM_NEW_SENTENCE, self.on_sentence)

        self.sentences_queue: Queue[str] = Queue()
        self.observable_sentences = queue_as_observable(self.sentences_queue)
        self.observable_sentences.subscribe(self.synthesise_sentence)

        self.tts = self.create_tts(PIPER_TTS_MODEL)
        self.is_synthesise = threading.Event()
        self.running = False
        logger.info("Synthesis ... IDLE")

    def create_tts(self, model: str):
        return PiperTts(model, PIPER_MODELS_LOCATION)

    def on_sentence(self, sentence: str):
        logger.info(f"> on_sentence({sentence})")
        self.sentences_queue.put(sentence)

    def on_interruption(self, iterrupt: Any):
        logger.warning(f"> on_interruption({iterrupt})")
        logger.info(f"Sentences to synth: {self.sentences_queue.qsize()}")

    def synthesise_sentence(self, sentence: str):
        with controlled_area(partial(self.event_bus.publish, EventType.SPEECH_SYNTH_STATUS), "running", "done", True, __name__):
            self.is_synthesise.set()
            speech, samplerate = self.tts.synthesize_stream(sentence)
            speech = enrich_with_silence(speech, samplerate, 0.1, 0.1)
            speech_resampled = self.resample_speec_for_mumble(speech, samplerate)

            self.event_bus.publish(EventType.MUMBLE_PLAY_AUDIO, speech_resampled)

            # Always clear state, after synth finished or interrupted.
            self.is_synthesise.clear()

    @staticmethod
    def resample_speec_for_mumble(speech: np.ndarray[np.int16], samplerate: int) -> np.ndarray[np.int16]:
        return resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)

    def run(self):
        self.running = True
        logger.info("Synthesis ... OK")

    def stop(self):
        logger.info("Synthesis ... STOPPING")
        self.running = False
        logger.info("Synthesis ... DEAD")
