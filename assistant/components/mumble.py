import logging
import numpy as np
from numpy.typing import NDArray
from queue import Queue
from typing import Any, Optional
import threading
from pymumble_py3 import Mumble
from pymumble_py3.callbacks import PYMUMBLE_CLBK_SOUNDRECEIVED, PYMUMBLE_CLBK_CONNECTED, PYMUMBLE_CLBK_DISCONNECTED
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk
from .event_bus import EventBus, EventType
from .synthesis import Sentence
from .util import queue_as_observable, controlled_area
from functools import partial

import reactivex as rx
from reactivex.abc import DisposableBase
from reactivex import operators as ops

from .audio import AudioBufferTransformer, chop_audio
from assistant.config import (
    ASSISTANT_NAME,
    MUMBLE_SERVER_HOST,
    MUMBLE_SERVER_PASSWORD,
    MUMBLE_SERVER_PORT,
    SPEECH_PIPELINE_SAMPLERATE,
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
)

logger = logging.getLogger(__name__)

# TODO: Create a generic interface to receive and play audio. Don't stick to Mumble.
class MumbleProcess:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.ignored_users = [ASSISTANT_NAME]
        self.client = self.create_client()

        self._buffer_transformer = AudioBufferTransformer(
            self.on_sound, SPEECH_PIPELINE_SAMPLERATE, SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
        )

        self.running = False
        self.connected: threading.Event = threading.Event()
        self.__is_playing = threading.Event()
        self.__is_interrupted = threading.Event()
        self.__is_play_done = threading.Event()

        self.play_audio_queue = Queue()
        self.observable_audios = queue_as_observable(self.play_audio_queue)
        self.observable_audios.subscribe(self.__on_play)
        self.playing_sub: Optional[DisposableBase] = None

        self.event_bus.subscribe(EventType.MUMBLE_PLAY_AUDIO, self.on_play)
        # NOTE: Not implemented from other side. See interruption manager TODO.
        self.event_bus.subscribe(EventType.MUMBLE_INTERRUPT_AUDIO, self.on_interrupt)
        logger.info("Mumble ... IDLE")

    @staticmethod
    def create_client():
        return Mumble(
            host=MUMBLE_SERVER_HOST,
            user=ASSISTANT_NAME,
            port=MUMBLE_SERVER_PORT,
            password=MUMBLE_SERVER_PASSWORD,
        )

    def get_number_of_items_to_play(self) -> int:
        return self.play_audio_queue.qsize()

    def is_alive(self) -> bool:
        return self.client.is_alive()

    def is_connected(self) -> bool:
        return self.connected.is_set()

    def is_ready(self):
        self.client.is_ready()

    def is_playing(self) -> bool:
        return self.__is_playing.is_set()

    def is_interrupted(self) -> bool:
        return self.__is_interrupted.is_set()

    def on_connect(self):
        self.connected.set()

    def on_disconnect(self):
        self.connected.clear()

    def on_interrupt(self, a: Any):
        logger.warning(f"on_interruption({a})")
        if self.__is_playing.is_set():
            self.__is_interrupted.set()

            if self.playing_sub is not None:
                self.playing_sub.dispose()

    def __on_interrupted(self):
        if self.__is_interrupted.is_set():
            while not self.play_audio_queue.empty():
                self.play_audio_queue.get()
                self.play_audio_queue.task_done()

            self.__is_interrupted.clear()
            self.__is_playing.clear()
            self.__is_play_done.set()

    def __on_play_complete(self):
        self.__is_playing.clear()
        self.__is_play_done.set()

    def on_play(self, sentence: Sentence):
        logger.info(f"> on_play('{sentence.text}')")
        self.play_audio_queue.put(sentence.audio)

    def __on_play(self, audio: NDArray[np.int16]):
        with controlled_area(partial(self.event_bus.publish, EventType.MUMBLE_PLAYING_STATUS), "running", "done", True, __name__):
            self.__is_play_done.clear()
            self.__is_playing.set()

            self.playing_sub = rx.zip(rx.interval(0.020), rx.from_iterable(chop_audio(audio, PYMUMBLE_SAMPLERATE, 20))).pipe(
                ops.map(lambda x: x[1].tobytes()),
                ops.do_action(self.client.sound_output.add_sound),
                ops.finally_action(self.__on_interrupted),
            ).subscribe(on_completed=self.__on_play_complete)

            self.__is_play_done.wait()

    def on_sound(self, sound: NDArray[np.float32]):
        logger.debug(f"{type(sound)}, {sound.flags.writeable}")
        self.event_bus.publish(EventType.MUMBLE_NEW_AUDIO, sound)

    def _sound_received_callback(self, user: str, soundchunk: SoundChunk):
        if user in self.ignored_users:
            logger.info(f"Sound from '{user}' was ignored. Because of ignore list.")
            return

        self._buffer_transformer(soundchunk)

    def run(self):
        self.running = True
        self.client.callbacks.set_callback(PYMUMBLE_CLBK_SOUNDRECEIVED, self._sound_received_callback)
        self.client.callbacks.set_callback(PYMUMBLE_CLBK_CONNECTED, self.on_connect)
        self.client.callbacks.set_callback(PYMUMBLE_CLBK_DISCONNECTED, self.on_disconnect)

        self.client.set_receive_sound(True)
        self.client.start()
        self.client.is_ready()
        logger.info("Mumble ... OK")

    def stop(self):
        logger.info("Mumble ... STOPPING")
        self.running = False
        self.client.stop()
        logger.info("Mumble ... DEAD")
