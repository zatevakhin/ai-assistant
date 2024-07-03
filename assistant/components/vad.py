from numpy.typing import NDArray
from voice_pulse import ListenerStamped, Config, VadEngine
from voice_pulse.input_sources import CallbackInput
import numpy as np
import threading
import logging
from .event_bus import EventBus
from reactivex.scheduler import NewThreadScheduler
import reactivex.operators as ops
import reactivex as rx
from functools import partial

from assistant.config import (
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    VAD_SILENCE_THRESHOLD,
    TOPIC_MUMBLE_SOUND_NEW,
    TOPIC_VAD_SPEECH_NEW
)

logger = logging.getLogger(__name__)

class VadProcess:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.config = Config(
            vad_engine=VadEngine.SILERIO,
            block_duration=SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
            silence_threshold=VAD_SILENCE_THRESHOLD,
        )
        self.stream = CallbackInput(self.config.blocksize)

        self.vad_subscription = self.event_bus.subscribe(TOPIC_MUMBLE_SOUND_NEW, self.on_sound)
        self.vad_scheduler = NewThreadScheduler()

        self.vad_observable = rx.from_iterable(ListenerStamped(self.config, self.stream)).pipe(
            ops.finally_action(self.on_stop),
            ops.subscribe_on(self.vad_scheduler)
        )

        on_speech = partial(self.event_bus.publish, TOPIC_VAD_SPEECH_NEW)
        self.vad_on_speech_subscription = self.vad_observable.subscribe(on_speech)

        logger.info("VAD Process ... IDLE")

    def on_stop(self):
        pass

    def on_sound(self, sound: NDArray[np.float32]):
        logger.debug(f"> on_sound({type(sound)})")
        self.stream.receive_chunk(sound)

    def run(self):
        logger.info("VAD Process ... OK")

    def stop(self):
        logger.info("VAD Process ... STOPPING")
        self.vad_subscription.dispose()
        self.vad_on_speech_subscription.dispose()
        self.stream.receive_chunk(None)

        logger.info("VAD Process ... DEAD")
