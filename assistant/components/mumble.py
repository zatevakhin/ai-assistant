import zenoh
from zenoh import Sample
import time
import logging
import numpy as np
from numpy.typing import NDArray
from queue import Queue, Empty
from typing import Any, Tuple, Optional, Union
import threading
from pymumble_py3 import Mumble
from pymumble_py3.callbacks import PYMUMBLE_CLBK_SOUNDRECEIVED, PYMUMBLE_CLBK_CONNECTED, PYMUMBLE_CLBK_DISCONNECTED
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk
from .event_bus import EventBus

import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import NewThreadScheduler
from reactivex import operators as ops

from .audio import AudioBufferTransformer, audio_length, chop_audio
from assistant.config import (
    ASSISTANT_NAME,
    MUMBLE_SERVER_HOST,
    MUMBLE_SERVER_PASSWORD,
    MUMBLE_SERVER_PORT,
    SPEECH_PIPELINE_SAMPLERATE,
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    TOPIC_MUMBLE_SOUND_NEW,
    TOPIC_MUMBLE_PLAY_AUDIO,
    TOPIC_MUMBLE_INTERRUPT_AUDIO,
    ZENOH_CONFIG
)

logger = logging.getLogger(__name__)

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
        self.is_playing = threading.Event()
        self.is_interrupted = threading.Event()

        self.playing_audio_subscribers: Queue[Disposable] = Queue()
        self.play_audio_scheduler = NewThreadScheduler()

        self.zenoh_session = zenoh.open(ZENOH_CONFIG)
        self.event_bus.subscribe(TOPIC_MUMBLE_PLAY_AUDIO, self.on_play)
        self.event_bus.subscribe(TOPIC_MUMBLE_INTERRUPT_AUDIO, self.on_interruption)

        self.pub_mumble_sound = self.zenoh_session.declare_publisher(TOPIC_MUMBLE_SOUND_NEW)
        self.sub_play_audio = self.zenoh_session.declare_subscriber(TOPIC_MUMBLE_PLAY_AUDIO, self.on_play)
        logger.info("Mumble ... IDLE")

    @staticmethod
    def create_client():
        return Mumble(
            host=MUMBLE_SERVER_HOST,
            user=ASSISTANT_NAME,
            port=MUMBLE_SERVER_PORT,
            password=MUMBLE_SERVER_PASSWORD,
        )

    def is_alive(self) -> bool:
        return self.client.is_alive()

    def is_connected(self) -> bool:
        return self.connected.is_set()

    def is_ready(self):
        self.client.is_ready()

    def on_connect(self):
        self.connected.set()

    def on_disconnect(self):
        self.connected.clear()

    def on_interruption(self, a: Any):
        logger.warning(f"on_interruption({a})")

        if self.is_playing.is_set() and self.playing_audio_subscribers.qsize() > 0:
            self.is_interrupted.set()
            self.playing_audio_subscribers.get().dispose()

    def on_play_interrupted(self):
        if self.is_interrupted.is_set():
            self.is_interrupted.wait()

            if self.playing_audio_subscribers.qsize() > 0:
                self.playing_audio_subscribers.get().dispose()

            self.is_interrupted.clear()
            self.is_playing.clear()

    def on_play_complete(self):
        self.is_playing.wait()

        if self.playing_audio_subscribers.qsize() > 0:
            self.playing_audio_subscribers.get().dispose()

        self.is_playing.clear()

    def on_play(self, audio: Union[NDArray[np.int16], Sample]) -> rx.Observable:
        logger.info(f"> on_play({type(audio)})")

        if isinstance(audio, Sample):
            audio = np.frombuffer(audio.payload, dtype=np.int16)

        # BUG: Can play multiple audios if on_play called multiple times.
        #      This happens on audio generation when sentence is split into pieces.
        observable_audio = rx.zip(rx.interval(0.020), rx.from_iterable(chop_audio(audio, PYMUMBLE_SAMPLERATE, 20))).pipe(
            ops.map(lambda x: x[1].tobytes()),
            ops.finally_action(self.on_play_interrupted),
            ops.subscribe_on(self.play_audio_scheduler)
        )

        subs = observable_audio.subscribe(self.on_play_chunk, on_error=lambda x: print(f"err: {x}"), on_completed=self.on_play_complete)
        self.playing_audio_subscribers.put(subs)
        #self.playing_audio_length_milis = audio_length(audio, PYMUMBLE_SAMPLERATE)

        self.is_playing.set()
        return observable_audio

    def on_play_chunk(self, audio_chunk: NDArray[np.int16]):
        self.client.sound_output.add_sound(audio_chunk)

    def on_sound(self, sound: np.ndarray[np.float32]):
        logger.debug(f"{type(sound)}, {sound.flags.writeable}")
        self.event_bus.publish(TOPIC_MUMBLE_SOUND_NEW, sound)

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
        time.sleep(3)
        logger.info("Mumble ... OK")

    def stop(self):
        logger.info("Mumble ... STOPPING")
        self.running = False
        self.client.stop()
        self.pub_mumble_sound.undeclare()
        self.sub_play_audio.undeclare()
        self.zenoh_session.close()
        self.is_playing.set()
        logger.info("Mumble ... DEAD")

def main():
    pass

if __name__ == "__main__":
    main()
