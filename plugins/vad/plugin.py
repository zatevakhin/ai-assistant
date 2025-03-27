from functools import partial
from reactivex.scheduler import NewThreadScheduler
from assistant.core import Plugin
from typing import Dict, Final, List
import reactivex as rx
from reactivex import operators as ops
from voice_pulse import ListenerStamped, Config, VadEngine, VadAggressiveness, SpeechSegment
from voice_pulse.input_sources import CallbackInput

from assistant.config import (
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    SPEECH_PIPELINE_SAMPLERATE,
    VAD_SILENCE_THRESHOLD,
)
from plugins.mumble.plugin import AudioChunk

from . import events
from plugins.mumble.events import MUMBLE_AUDIO_CHUNK


STREAM_CONFIG: Final = Config(
    device=None,
    channels=1,
    samplerate=SPEECH_PIPELINE_SAMPLERATE,
    vad_engine=VadEngine.SILERIO,
    vad_aggressiveness=VadAggressiveness.LOW,
    block_duration=SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    silence_threshold=VAD_SILENCE_THRESHOLD,
    collect_threshold=8
)


class VadFilter(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.VAD_SPEECH_DETECT,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.logger.setLevel(self.get_config("log_level", "INFO"))
        self.add_dependency("mumble")

        # FIXME: Memleak if works too long. Need cleanup.
        self.streams: Dict[str, CallbackInput] = {}

        sub = self.event_bus.subscribe(MUMBLE_AUDIO_CHUNK, self.on_sound)
        self.subscriptions.append(sub)

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def get_or_create_stream_handler(self, source: str):
        if source in self.streams:
            return self.streams[source]

        self.streams[source] = CallbackInput(STREAM_CONFIG.blocksize)

        sub = rx.from_iterable(ListenerStamped(STREAM_CONFIG, self.streams[source])).pipe(
            ops.finally_action(partial(self.on_stop, source)),
            ops.subscribe_on(NewThreadScheduler())
        ).subscribe(partial(self.on_speech_detect_from_source, source))

        self.subscriptions.append(sub)
        return self.streams[source]



    def shutdown(self) -> None:
        super().shutdown()
        # TODO: Fix type hints in `voice_pulse`
        for (_, stream) in self.streams.items():
            stream.receive_chunk(None)
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_stop(self, source: str):
        self.logger.warning(f"Plugin '{self.name}' listener for source '{source}' was stopped.")

    def on_sound(self, sound: AudioChunk):
        self.logger.debug(f"> on_sound({sound})")

        stream = self.get_or_create_stream_handler(sound.source)
        stream.receive_chunk(sound.data)

    def on_speech_detect(self, speech: SpeechSegment):
        self.logger.info(f"> on_speech_detect({type(speech)})")
        self.event_bus.publish(events.VAD_SPEECH_DETECT, speech)

    def on_speech_detect_from_source(self, source: str, speech: SpeechSegment):
        self.logger.info(f"> on_speech_detect('{source}', {type(speech)})")
        # TODO: Add custom type for speech with source
        self.event_bus.publish(events.VAD_SPEECH_DETECT, (source, speech))


