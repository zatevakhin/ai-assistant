from reactivex.scheduler import NewThreadScheduler
from assistant.core import Plugin
from typing import List
from numpy.typing import NDArray
import numpy as np
import reactivex as rx
from reactivex import operators as ops
from voice_pulse import ListenerStamped, Config, VadEngine, VadAggressiveness, SpeechSegment
from voice_pulse.input_sources import CallbackInput

from assistant.config import (
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    VAD_SILENCE_THRESHOLD,
)

from . import events
from plugins.mumble.events import MUMBLE_AUDIO_CHUNK


class VadFilter(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.VAD_SPEEC_DETECT,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.add_dependency("mumble")

        self.config = Config(
            device=None,
            channels=1,
            samplerate=16000,
            vad_engine=VadEngine.SILERIO,
            vad_aggressiveness=VadAggressiveness.LOW,
            block_duration=SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
            silence_threshold=VAD_SILENCE_THRESHOLD,
            collect_threshold=8
        )
        self.stream = CallbackInput(self.config.blocksize)
        sub = self.event_bus.subscribe(MUMBLE_AUDIO_CHUNK, self.on_sound)
        self.subscriptions.append(sub)

        sub1 = rx.from_iterable(ListenerStamped(self.config, self.stream)).pipe(
            ops.finally_action(self.on_stop),
            ops.subscribe_on(NewThreadScheduler())
        ).subscribe(self.on_speech_detect)

        self.subscriptions.append(sub1)

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        # TODO: Fix type hints in `voice_pulse`
        self.stream.receive_chunk(None)
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_stop(self):
        self.logger.warning(f"Plugin '{self.name}' scheduler stopped.")

    def on_sound(self, sound: NDArray[np.float32]):
        self.logger.debug(f"> on_sound({type(sound)})")
        self.stream.receive_chunk(sound)

    def on_speech_detect(self, speech: SpeechSegment):
        self.logger.info(f"> on_speech_detect({type(speech)})")
        self.event_bus.publish(events.VAD_SPEEC_DETECT, speech)
