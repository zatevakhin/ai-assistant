import logging
import threading
from queue import Queue
import resampy
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from typing import Any

from assistant.config import (
    PIPER_TTS_MODEL,
    PIPER_MODELS_LOCATION,
    TOPIC_LLM_ON_SENTENCE,
    TOPIC_MUMBLE_PLAY_AUDIO,
)

from voice_forge import PiperTts
from .util import queue_as_observable
import numpy as np
from .event_bus import EventBus

logger = logging.getLogger(__name__)


class SpeechSynthesisProcess:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.sentence_subscription = self.event_bus.subscribe(TOPIC_LLM_ON_SENTENCE, self.on_sentence)
        # self.interrupt_subscription = self.event_bus.subscribe(TOPIC_SPEECH_SYNTHESIS_INTERRUPT, self.on_interruption)

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
        self.is_synthesise.set()
        speech, samplerate = self.tts.synthesize_stream(sentence)
        speech_resampled = self.resample_speec_for_mumble(speech, samplerate)

        self.event_bus.publish(TOPIC_MUMBLE_PLAY_AUDIO, speech_resampled)

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
